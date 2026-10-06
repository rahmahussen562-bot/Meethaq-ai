"""API/CORS regressions using an injected pipeline, without native database startup."""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

import api
from grounding import ABSTENTION


class FakePipeline:
    def stats(self):
        return {"indexed_chunks": 0, "calibration_status": "required", "answer_mode": "extractive",
                "llm_invoked": False, "collection": "meethaq_contracts",
                "embed_model": "ONNX", "rerank_model": "cosine", "llm_model": "llama3.2:3b"}

    def audit(self, question, expand_query=False):
        return {"answer": ABSTENTION, "sources": [], "stats": self.stats()}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = FakePipeline()
        api.app.dependency_overrides[api.get_rag] = lambda: self.pipeline
        self.client = TestClient(api.app)

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def test_local_and_cloudflare_origins_are_allowed(self):
        for origin in [
            "http://localhost:8443", "http://127.0.0.1:8443", "https://my-app.pages.dev",
            "https://preview.my-app.pages.dev", "https://meethaq.workers.dev",
            "https://random-words.trycloudflare.com",
        ]:
            with self.subTest(origin=origin):
                response = self.client.options("/api/audit", headers={
                    "Origin": origin, "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-allow-origin"], origin)
                self.assertNotIn("access-control-allow-credentials", response.headers)

    def test_crafted_hostnames_and_null_origins_are_rejected(self):
        for origin in [
            "https://pages.dev.evil.example", "https://evilpages.dev",
            "https://myapp.pages.dev@evil.example", "https://localhost.evil.example",
            "https://evil.example/allowed.pages.dev", "null",
        ]:
            with self.subTest(origin=origin):
                response = self.client.options("/api/audit", headers={
                    "Origin": origin, "Access-Control-Request-Method": "POST",
                })
                self.assertEqual(response.status_code, 400)
                self.assertNotIn("access-control-allow-origin", response.headers)

    def test_truthful_zero_telemetry_and_abstention(self):
        telemetry = self.client.get("/api/telemetry")
        self.assertEqual(telemetry.status_code, 200)
        self.assertEqual(telemetry.json()["total_chunks"], 0)
        self.assertEqual(telemetry.json()["indexed_chunks"], 0)
        response = self.client.post("/api/audit", json={"query": "What governing law applies?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], ABSTENTION)
        self.assertEqual(response.json()["sources"], [])

    def test_health_reports_readiness_separately_from_reachability(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])
        self.assertEqual(response.json()["status"], "not_ready")
        with mock.patch.object(self.pipeline, "stats", return_value={
            **self.pipeline.stats(), "indexed_chunks": 12, "calibration_status": "calibrated"
        }):
            response = self.client.get("/api/health")
            self.assertTrue(response.json()["ready"])
            self.assertEqual(response.json()["status"], "ready")

    def test_invalid_inputs_are_rejected(self):
        for query, status in [("", 422), ("  ", 400), ("x" * 2001, 422)]:
            response = self.client.post("/api/audit", json={"query": query})
            self.assertEqual(response.status_code, status)

    def test_store_failure_is_503_never_fake_zero(self):
        with mock.patch.object(self.pipeline, "stats", side_effect=RuntimeError("index unavailable")):
            with self.assertLogs("api", "ERROR"):
                response = self.client.get("/api/telemetry")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("total_chunks", response.json())

    def test_internal_retrieval_value_error_is_logged_as_backend_failure(self):
        with mock.patch.object(self.pipeline, "audit", side_effect=ValueError("invalid stored embeddings")):
            with self.assertLogs("api", "ERROR"):
                response = self.client.post("/api/audit", json={"query": "What governing law applies?"})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("invalid stored embeddings", response.json()["detail"])

    def test_shutdown_releases_cached_pipeline(self):
        resource = mock.Mock()
        with mock.patch.object(api, "_pipeline", resource):
            with TestClient(api.app):
                resource.close.assert_not_called()
            resource.close.assert_called_once()
            self.assertIsNone(api._pipeline)

    def test_additional_origins_require_explicit_http_origins(self):
        with mock.patch.dict("os.environ", {"MEETHAQ_CORS_ORIGINS": "https://contracts.example,http://localhost:9000"}):
            self.assertEqual(api.additional_origins(), ["https://contracts.example", "http://localhost:9000"])
        for invalid in ["*", "https://*.example", "https://x.example/path", "https://u:p@x.example"]:
            with mock.patch.dict("os.environ", {"MEETHAQ_CORS_ORIGINS": invalid}):
                with self.assertRaises(ValueError):
                    api.additional_origins()


if __name__ == "__main__":
    unittest.main()

