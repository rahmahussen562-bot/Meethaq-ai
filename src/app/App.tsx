import {
  Component,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react"
import {
  createBrowserRouter,
  Link,
  RouterProvider,
  useNavigate,
  useSearchParams,
} from "react-router"
import {
  DEFAULT_TELEMETRY,
  fetchTelemetry,
  sourceForCitation,
  sendAuditQuery,
  type AuditResponse,
  type SourceItem,
  type TelemetryResponse,
} from "../services/api"

export class ErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode; fallback?: ReactNode }) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("[Meethaq UI ErrorBoundary Caught]:", error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }
      return (
        <div className="flex min-h-screen items-center justify-center bg-[#fdfbf7] p-6 text-stone-800">
          <div className="w-full max-w-md rounded-2xl border border-[#e3ddd4] bg-white p-8 text-center shadow-sm">
            <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-full bg-[#f2e8dd] text-[#8b6141]">
              <svg
                className="size-6"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>
            <h2 className="font-serif text-2xl font-semibold mb-2">
              Meethaq AI Recovery Notice
            </h2>
            <p className="mb-6 text-xs leading-relaxed text-stone-500">
              The application encountered a recoverable view exception. The
              isolated state prevented an unhandled UI crash.
            </p>
            <div className="flex justify-center gap-3">
              <button
                type="button"
                onClick={() => {
                  this.setState({ hasError: false, error: null })
                  window.location.href = "/"
                }}
                className="cursor-pointer rounded-lg bg-stone-800 px-4 py-2.5 text-xs font-medium text-white hover:bg-stone-700"
              >
                Command Center
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="cursor-pointer rounded-lg border border-stone-300 px-4 py-2.5 text-xs font-medium hover:bg-stone-100"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

export function RouteErrorFallback() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#fdfbf7] p-6 text-stone-800">
      <div className="w-full max-w-md rounded-2xl border border-[#e3ddd4] bg-white p-8 text-center shadow-sm">
        <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-full bg-[#f2e8dd] text-[#8b6141]">
          <svg
            className="size-6"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </div>
        <h2 className="font-serif text-2xl font-semibold mb-2">
          Workspace Navigation Fallback
        </h2>
        <p className="mb-6 text-xs leading-relaxed text-stone-500">
          The requested route encountered a display issue. Return to the dashboard
          to proceed with your local legal audit.
        </p>
        <Link
          to="/"
          className="inline-flex cursor-pointer items-center justify-center rounded-lg bg-stone-800 px-5 py-2.5 text-xs font-medium text-white hover:bg-stone-700"
        >
          Return to Command Center
        </Link>
      </div>
    </div>
  )
}

type MeethaqContextValue = {
  searchQuery: string
  setSearchQuery: (value: string) => void
  auditResponse: AuditResponse | null
  selectedSource: SourceItem | null
  setSelectedSource: (source: SourceItem | null) => void
  isLoading: boolean
  errorMessage: string | null
  telemetry: TelemetryResponse
  runAudit: (query: string, expandQuery?: boolean) => Promise<void>
}

const MeethaqContext = createContext<MeethaqContextValue | null>(null)

function useMeethaq() {
  const ctx = useContext(MeethaqContext)
  if (!ctx) {
    throw new Error("useMeethaq must be used within MeethaqProvider")
  }
  return ctx
}

function MeethaqProvider({ children }: { children: ReactNode }) {
  const [searchQuery, setSearchQuery] = useState("")
  const [auditResponse, setAuditResponse] = useState<AuditResponse | null>(null)
  const [selectedSource, setSelectedSource] = useState<SourceItem | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetryResponse>(DEFAULT_TELEMETRY)
  const auditSequence = useRef(0)
  const telemetryVersion = useRef(0)
  const activeAudit = useRef<AbortController | null>(null)
  const lifecycleVersion = useRef(0)

  useEffect(() => {
    const mountedVersion = ++lifecycleVersion.current
    let disposed = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let poll: AbortController | undefined
    async function refreshTelemetry() {
      poll = new AbortController()
      const version = telemetryVersion.current
      try {
        const data = await fetchTelemetry(poll.signal)
        if (!disposed && version === telemetryVersion.current) setTelemetry(data)
      } catch (error) {
        if (!disposed && version === telemetryVersion.current) {
          setTelemetry({ status: error instanceof Error ? error.message : "Contract index unavailable." })
        }
      } finally {
        if (!disposed) timer = setTimeout(refreshTelemetry, 15000)
      }
    }
    void refreshTelemetry()
    return () => {
      disposed = true
      clearTimeout(timer)
      poll?.abort()
      // StrictMode immediately replays setup; cancel only if it stays unmounted.
      queueMicrotask(() => {
        if (lifecycleVersion.current === mountedVersion) {
          activeAudit.current?.abort()
          auditSequence.current += 1
        }
      })
    }
  }, [])

  const runAudit = useCallback(async (query: string, expandQuery = false) => {
    const question = typeof query === "string" ? query.trim() : ""
    if (!question) return
    activeAudit.current?.abort()
    const controller = new AbortController()
    activeAudit.current = controller
    const sequence = ++auditSequence.current
    telemetryVersion.current += 1
    setSearchQuery(question)
    setAuditResponse(null)
    setSelectedSource(null)
    setIsLoading(true)
    setErrorMessage(null)
    try {
      const result = await sendAuditQuery(question, expandQuery, controller.signal)
      if (sequence !== auditSequence.current) return
      setAuditResponse(result)
      setSelectedSource(result.sources[0] ?? null)
      telemetryVersion.current += 1
      setTelemetry({ ...result.stats, status: result.stats.status || "Contract index connected" })
    } catch (error) {
      if (sequence !== auditSequence.current || controller.signal.aborted) return
      setErrorMessage(error instanceof Error ? error.message : "The audit request failed.")
    } finally {
      if (sequence === auditSequence.current) setIsLoading(false)
    }
  }, [])

  return (
    <MeethaqContext.Provider value={{
      searchQuery, setSearchQuery, auditResponse, selectedSource,
      setSelectedSource, isLoading, errorMessage, telemetry, runAudit,
    }}>
      {children}
    </MeethaqContext.Provider>
  )
}

function chunkCountLabel(telemetry: TelemetryResponse): string {
  const count = telemetry.indexed_chunks ?? telemetry.total_chunks
  return typeof count === "number" ? count.toLocaleString() : "\u2014"
}

function Icon({
  name,
  className = "size-4",
}: {
  name: string
  className?: string
}) {
  const paths: Record<string, string> = {
    search: "M17 17l4 4M18 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z",
    arrow: "M4 12h16m-6-6 6 6-6 6",
    left: "m14 6-6 6 6 6",
    right: "m10 6 6 6-6 6",
    file: "M6 3h8l4 4v14H6V3Zm8 0v5h4M9 12h6m-6 4h6",
    table: "M3 4h18v16H3V4Zm0 5h18M9 9v11m6-11v11",
    chat: "M14 4H3v14h4v3l4-3h10v-7M18 2v6m-3-3h6",
    clip: "m8 12 7-7a3 3 0 0 1 4 4l-9 9a5 5 0 0 1-7-7l9-9m-6 12 8-8",
    check: "m5 12 4 4L19 6",
    shield: "m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6",
    pause: "M8 5v14M16 5v14",
    play: "m8 4 12 8-12 8V4Z",
    plus: "M12 5v14M5 12h14",
    more: "M5 12h1m5 0h1m5 0h1",
    folder: "M3 6h7l2 3h9v11H3V6Z",
    lock: "M7 10V7a5 5 0 0 1 10 0v3M5 10h14v11H5V10Zm7 4v3",
    close: "m6 6 12 12M6 18 18 6",
    download: "M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4",
    copy: "M9 8h12v13H9V8ZM5 16H3V3h12v2",
    alert: "m12 3 10 18H2L12 3Zm0 6v5m0 3v1",
  }
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] || paths.file} />
    </svg>
  )
}
const primary =
  "inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-xs font-medium text-white transition-colors hover:bg-stone-700"
const secondary =
  "inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-white/40 px-3.5 py-2.5 text-xs font-medium transition-colors hover:bg-muted"
const templates = [
  {
    title: "Vendor Agreement Summary & Conflict Check",
    count: 12,
    category: "MSA Audit",
    icon: "file",
    query: "Compare MSA §8 with Addendum 2 and summarize penalty conflicts.",
  },
  {
    title: "M&A Due Diligence & Liabilities",
    count: 18,
    category: "Risk Assessment",
    icon: "shield",
    query: "Review the documents for M&A due diligence and liability exposure.",
  },
  {
    title: "SLA Downtime & Penalty Calculation",
    count: 8,
    category: "Telecom & Cloud",
    icon: "table",
    query: "Compare SLA downtime obligations and calculate penalty exposure.",
  },
  {
    title: "Contract Renewal & Obligations",
    count: 10,
    category: "Contract Review",
    icon: "folder",
    query:
      "Identify renewal dates and ongoing obligations across the contracts.",
  },
]
const sessions = [
  {
    name: "Cloud Infrastructure MSA",
    subtitle: "3 documents · Vendor audit",
    time: "Just now",
    status: "Building Table...",
    type: "building",
    action: "Pause",
    second: "",
  },
  {
    name: "Vendor SLA Dispute (Q3)",
    subtitle: "2 documents · Cross-document audit",
    time: "2h ago",
    status: "1 Conflict Flagged",
    type: "conflict",
    action: "Chat with Results",
    second: "View Results",
  },
  {
    name: "Addendum No. 2 Liability Cap",
    subtitle: "1 document · Clause analysis",
    time: "Yesterday",
    status: "470 Chunks Indexed",
    type: "indexed",
    action: "Chat with Results",
    second: "View Proof",
  },
  {
    name: "Master Service Agreement 2026",
    subtitle: "4 documents · Agreement review",
    time: "Aug 20, 2026",
    status: "Completed",
    type: "complete",
    action: "View Chat",
    second: "Export Evidence",
  },
]

function Header({
  onSearch,
  onCreate,
}: {
  onSearch?: (value: string) => void
  onCreate?: () => void
}) {
  const { telemetry } = useMeethaq()
  const searchRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const [profile, setProfile] = useState(false)
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key?.toLowerCase() === "k") {
        event.preventDefault()
        searchRef.current?.focus()
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [])

  const chunkDisplay = chunkCountLabel(telemetry)

  return (
    <header className="flex h-[76px] items-center justify-between gap-5 border-b border-border px-5 md:px-9">
      <Link to="/" className="flex shrink-0 items-center gap-3">
        <span className="text-lg font-semibold tracking-[0.13em]">MEETHAQ</span>
        <span className="hidden border-l border-border pl-3 font-mono text-[9px] text-muted-foreground sm:block">
          v1.0 LOCAL
        </span>
        <span title={telemetry.status} className="hidden max-w-48 truncate border-l border-border pl-3 font-mono text-[9px] text-muted-foreground md:block">
          {telemetry.status || "Checking contract index..."}
        </span>
        <span className="hidden border-l border-border pl-3 font-mono text-[9px] text-muted-foreground lg:block">
          {chunkDisplay} CHUNKS
        </span>
      </Link>
      <div className="hidden w-full max-w-[460px] items-center gap-3 rounded-lg border border-border bg-white/50 px-3 py-2.5 lg:flex">
        <Icon name="search" className="size-4 shrink-0 text-stone-500" />
        <input
          ref={searchRef}
          aria-label="Search workspaces"
          placeholder="Search contracts, clauses, or citations..."
          onChange={(event) => onSearch?.(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              event.currentTarget.value.trim() &&
              !onSearch
            )
              navigate(
                `/?search=${encodeURIComponent(event.currentTarget.value)}`,
              )
          }}
          className="min-w-0 flex-1 bg-transparent text-[11px] outline-none placeholder:text-stone-400"
        />
        <kbd className="whitespace-nowrap rounded border border-border bg-background px-1.5 py-0.5 font-sans text-[9px] text-stone-400">
          Ctrl + K
        </kbd>
      </div>
      <div className="relative flex shrink-0 items-center gap-2.5">
        <button
          className={`${secondary} hidden sm:flex`}
          onClick={() => (onCreate ? onCreate() : navigate("/?create=table"))}
        >
          <Icon name="table" />
          Create Table
        </button>
        <button className={primary} onClick={() => navigate("/audit")}>
          <Icon name="chat" />
          Ask Meethaq
        </button>
        <button
          aria-label="User profile"
          aria-expanded={profile}
          onClick={() => setProfile(!profile)}
          className="ml-2 flex size-9 items-center justify-center rounded-full border border-[#ddd5cb] bg-[#eee8df] font-serif text-sm"
        >
          AK
        </button>
        {profile && (
          <div className="absolute right-0 top-12 z-20 w-56 rounded-lg border border-border bg-white p-4 shadow-sm">
            <p className="text-xs font-medium">Ahmed K.</p>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Enterprise workspace administrator
            </p>
            <p className="mt-4 border-t border-border pt-3 font-mono text-[9px] text-muted-foreground">
              LOCAL SESSION · PRIVATE
            </p>
          </div>
        )}
      </div>
    </header>
  )
}

function Dashboard() {
  const { searchQuery, setSearchQuery, telemetry, runAudit, errorMessage } = useMeethaq()
  const [params] = useSearchParams()
  const [search, setSearch] = useState(params.get("search") || "")
  const [offset, setOffset] = useState(0)
  const [all, setAll] = useState(false)
  const [paused, setPaused] = useState(false)
  const [menu, setMenu] = useState<number | null>(null)
  const [creating, setCreating] = useState(params.get("create") === "table")
  const [tableName, setTableName] = useState("")
  const [custom, setCustom] = useState<string[]>([])
  const [attachment, setAttachment] = useState("")
  const [tab, setTab] = useState("All workspaces")
  const fileRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const visibleTemplates = all
    ? templates
    : Array.from(
        { length: 3 },
        (_, index) => templates[(index + offset) % templates.length],
      )
  const filtered = sessions.filter(
    (session) =>
      `${session.name} ${session.subtitle} ${session.status}`
        .toLowerCase()
        .includes(search.toLowerCase()) &&
      (tab !== "Needs attention" || session.type === "conflict"),
  )
  function audit(value: string) {
    void runAudit(value)
    navigate(`/audit?q=${encodeURIComponent(value)}`)
  }
  return (
    <div className="min-h-screen">
      <Header onSearch={setSearch} onCreate={() => setCreating(true)} />
      <main className="mx-auto max-w-[1320px] px-5 pb-16 pt-12 md:px-12 md:pt-14">
        <div className="mb-5 flex items-center gap-2 font-mono text-[9px] tracking-[0.12em] text-muted-foreground">
          <span className="size-1.5 rounded-full bg-[#687867]" />
          PRIVATE INTELLIGENCE<span className="mx-1 text-stone-300">/</span>YOUR
          COMMAND CENTER
        </div>
        <div className="flex items-end justify-between gap-6">
          <div>
            <h1 className="font-serif text-[38px] leading-[1.15] font-normal md:text-[48px]">
              Meethaq Legal Command Center
            </h1>
            <p className="mt-3 text-[13px] leading-6 text-muted-foreground">
              Deterministic contract intelligence powered by local air-gapped
              RAG.
            </p>
          </div>
          <div className="hidden pb-1 text-right md:block">
            <p className="font-mono text-[10px] text-stone-500">Q3 2026</p>
            <p className="mt-2 text-[11px] text-stone-400">
              Enterprise Vendor Audit
            </p>
          </div>
        </div>
        <section className="mt-11">
          <div className="mb-4 flex items-center justify-between gap-4">
            <h2 className="text-[13px] font-medium">
              Start with an Audit Workflow Template
            </h2>
            <div className="flex items-center gap-2">
              <button
                aria-label="Previous templates"
                onClick={() =>
                  setOffset((offset + templates.length - 1) % templates.length)
                }
                className="flex size-7 items-center justify-center rounded-full border border-border hover:bg-muted"
              >
                <Icon name="left" className="size-3.5" />
              </button>
              <button
                aria-label="Next templates"
                onClick={() => setOffset((offset + 1) % templates.length)}
                className="flex size-7 items-center justify-center rounded-full border border-border hover:bg-muted"
              >
                <Icon name="right" className="size-3.5" />
              </button>
              <button
                className="ml-3 text-[11px] text-stone-600 hover:text-black"
                onClick={() => setAll(!all)}
              >
                {all ? "Show Less" : "View All"}
              </button>
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            {visibleTemplates.map((template) => (
              <button
                key={template.title}
                onClick={() => audit(template.query)}
                className="group flex min-h-[176px] flex-col rounded-lg border border-border bg-muted px-5 py-5 text-left transition-colors hover:border-stone-400 hover:bg-[#eee9e1]"
              >
                <div className="flex items-center justify-between">
                  <Icon
                    name={template.icon}
                    className="size-5 text-stone-600"
                  />
                  <span className="flex items-center gap-1.5 rounded-full border border-[#e3ddd4] px-2 py-1 font-mono text-[9px] text-stone-500">
                    <Icon name="check" className="size-2.5" />
                    {template.count} Prompts
                  </span>
                </div>
                <h3 className="mt-5 max-w-[270px] text-[14px] leading-[1.6] font-medium">
                  {template.title}
                </h3>
                <div className="mt-auto flex items-center justify-between pt-4 text-[10px] text-stone-500">
                  <span>{template.category}</span>
                  <Icon
                    name="arrow"
                    className="size-4 transition-transform group-hover:translate-x-1"
                  />
                </div>
              </button>
            ))}
          </div>
        </section>
        <section className="mt-9">
          <form
            onSubmit={(event) => {
              event.preventDefault()
              if (searchQuery.trim()) audit(searchQuery)
            }}
            className="rounded-xl border border-[#e3ddd4] bg-white p-4 shadow-[0_3px_16px_0_rgba(48,40,27,0.035)]"
          >
            <div className="flex items-start gap-3">
              <span className="mt-1 text-stone-400">
                <Icon name="chat" className="size-5" />
              </span>
              <textarea
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                aria-label="Compliance question"
                placeholder="Enter a compliance question, paste a clause, or compare documents (e.g. Compare MSA §8 with Addendum 2)..."
                className="min-h-[52px] w-full resize-none bg-transparent pt-1 text-xs leading-6 outline-none placeholder:text-stone-400"
              />
            </div>
            <div className="mt-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  aria-label="Select a document filename (not indexed)"
                  onClick={() => fileRef.current?.click()}
                  className="rounded-md p-1.5 text-stone-500 hover:bg-muted"
                >
                  <Icon name="clip" className="size-5" />
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.txt,.docx"
                  className="hidden"
                  onChange={(event) =>
                    setAttachment(event.target.files?.[0]?.name || "")
                  }
                />
                {attachment ? (
                  <span className="text-[10px] text-stone-500">
                    {attachment} - Not indexed; run local ingestion first
                  </span>
                ) : (
                  <span className="hidden text-[10px] text-stone-400 sm:block">
                    Grounded in your documents. Built for your judgment.
                  </span>
                )}
              </div>
              <button className={primary} disabled={!searchQuery.trim()}>
                Run Audit
                <Icon name="arrow" className="size-3.5" />
              </button>
            </div>
          </form>
          <p className="mt-3 flex items-center justify-center gap-1.5 font-mono text-[8px] tracking-wide text-stone-400">
            <Icon name="lock" className="size-2.5" />
            AIR-GAPPED BY DESIGN<span className="mx-1">·</span>ZERO CLOUD LEAK
            <span className="mx-1">·</span>TRACEABLE EVIDENCE
          </p>
        </section>
        <section className="mt-11">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h2 className="font-serif text-[29px]">Sample Workspaces</h2>
              <span className="rounded-full border border-border px-2 py-0.5 font-mono text-[9px] text-stone-500">
                {sessions.length + custom.length}
              </span>
            </div>
            <button
              className="flex items-center gap-2 text-[11px] text-stone-600"
              onClick={() => setCreating(true)}
            >
              <Icon name="plus" className="size-3.5" />
              New workspace
            </button>
          </div>
          <p className="mt-2 text-[10px] text-stone-500">Sample sessions for navigation. Index status is reported by the backend above.</p>
          <div className="mt-4 flex items-center justify-between border-b border-border">
            <div className="flex gap-6">
              {["All workspaces", "Needs attention"].map((label) => (
                <button
                  key={label}
                  onClick={() => setTab(label)}
                  className={`border-b py-3 text-[11px] ${
                    tab === label
                      ? "border-stone-800 font-medium"
                      : "border-transparent text-stone-400"
                  }`}
                >
                  {label}
                  {label === "Needs attention" && (
                    <span className="ml-2 rounded-full bg-[#ede5db] px-1.5 py-0.5 text-[8px] text-stone-600">
                      1
                    </span>
                  )}
                </button>
              ))}
            </div>
            <span className="hidden text-[10px] text-stone-400 sm:block">
              Last updated
              <Icon name="left" className="ml-2 inline size-3 -rotate-90" />
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[810px] text-left">
              <thead className="text-[9px] text-stone-400">
                <tr>
                  <th className="py-4 font-normal">Workspace name</th>
                  <th className="w-[13%] font-normal">Last activity</th>
                  <th className="w-[24%] font-normal">Status</th>
                  <th className="w-[26%] text-right font-normal">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((session) => {
                  const index = sessions.indexOf(session)
                  return (
                    <tr key={session.name} className="border-t border-border">
                      <td className="py-5">
                        <div className="flex items-center gap-3">
                          <span className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-border bg-white/50 text-stone-500">
                            <Icon
                              name={
                                session.type === "building" ? "table" : "folder"
                              }
                              className="size-4"
                            />
                          </span>
                          <div>
                            <button
                              onClick={() => audit(session.name)}
                              className="text-[12px] font-medium hover:underline"
                            >
                              {session.name}
                            </button>
                            <p className="mt-1 text-[9px] text-stone-400">
                              {session.subtitle}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="text-[10px] text-stone-500">
                        {session.time}
                      </td>
                      <td>
                        {session.type === "building" ? (
                          <div className="max-w-[170px]">
                            <div className="flex justify-between text-[10px] text-stone-500">
                              <span>
                                {paused ? "Table paused" : session.status}
                              </span>
                              <span className="font-mono text-[9px]">45%</span>
                            </div>
                            <div className="mt-2 h-[3px] rounded-full bg-[#e8e3db]">
                              <div className="h-full w-[45%] rounded-full bg-stone-600" />
                            </div>
                          </div>
                        ) : (
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2 py-1.5 text-[9px] ${
                              session.type === "conflict"
                                ? "bg-[#f2e8dd] text-[#8b6141]"
                                : session.type === "complete"
                                  ? "bg-[#e9eee6] text-[#5c6c53]"
                                  : "bg-muted text-stone-600"
                            }`}
                          >
                            <Icon
                              name={
                                session.type === "conflict"
                                  ? "alert"
                                  : session.type === "complete"
                                    ? "check"
                                    : "file"
                              }
                              className="size-3"
                            />
                            <span
                              className={
                                session.type === "indexed"
                                  ? "font-mono text-[8px]"
                                  : ""
                              }
                            >
                              {session.status}
                            </span>
                          </span>
                        )}
                      </td>
                      <td>
                        <div className="flex items-center justify-end gap-3">
                          <button
                            onClick={() =>
                              session.type === "building"
                                ? setPaused(!paused)
                                : audit(session.name)
                            }
                            className={
                              session.type === "conflict" ||
                              session.type === "indexed"
                                ? "whitespace-nowrap rounded-full bg-primary px-3 py-2 text-[9px] text-white hover:bg-stone-700"
                                : "flex items-center gap-1.5 rounded-full border border-border px-3 py-2 text-[9px] hover:bg-muted"
                            }
                          >
                            {session.type === "building" && (
                              <Icon
                                name={paused ? "play" : "pause"}
                                className="size-3"
                              />
                            )}
                            {session.type === "building" && paused
                              ? "Resume"
                              : session.action}
                          </button>
                          {session.second && (
                            <button
                              onClick={() =>
                                session.second === "Export Evidence"
                                  ? navigate("/audit")
                                  : navigate(
                                      `/audit?q=${encodeURIComponent(session.name)}&view=evidence`,
                                    )
                              }
                              className="whitespace-nowrap text-[9px] text-stone-500 hover:text-black"
                            >
                              {session.second}
                            </button>
                          )}
                          <div className="relative">
                            <button
                              aria-label={`More options for ${session.name}`}
                              aria-expanded={menu === index}
                              onClick={() =>
                                setMenu(menu === index ? null : index)
                              }
                              className="rounded p-1 hover:bg-muted"
                            >
                              <Icon name="more" className="size-4" />
                            </button>
                            {menu === index && (
                              <div className="absolute right-0 top-7 z-10 w-36 rounded-lg border border-border bg-white p-1 shadow-sm">
                                <button
                                  onClick={() => audit(session.name)}
                                  className="w-full px-3 py-2 text-left text-[10px] hover:bg-muted"
                                >
                                  Open workspace
                                </button>
                                <button
                                  onClick={() => navigate("/audit")}
                                  className="w-full px-3 py-2 text-left text-[10px] hover:bg-muted"
                                >
                                  Export summary
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )
                })}
                {tab === "All workspaces" &&
                  custom
                    .filter((name) =>
                      name.toLowerCase().includes(search.toLowerCase()),
                    )
                    .map((name) => (
                      <tr key={name} className="border-t border-border">
                        <td className="py-5 text-xs">{name}</td>
                        <td className="text-[10px] text-stone-500">Just now</td>
                        <td className="text-[10px] text-stone-500">
                          Ready for documents
                        </td>
                        <td className="text-right">
                          <button
                            className={secondary}
                            onClick={() => audit(name)}
                          >
                            Open workspace
                          </button>
                        </td>
                      </tr>
                    ))}
              </tbody>
            </table>
            {!filtered.length && !custom.length && (
              <p className="py-10 text-center text-xs text-stone-500">
                No matching workspaces. Try a different search.
              </p>
            )}
          </div>
        </section>
        <footer className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-5 text-[9px] text-stone-400">
          <span>
            Meethaq AI · Private & deterministic contract intelligence
          </span>
          <span className="flex items-center gap-1.5 font-mono text-[8px]">
            <span className="size-1 rounded-full bg-[#687867]" />
            LOCAL RUNTIME<span className="mx-2">/</span>
            {chunkCountLabel(telemetry)} CHUNKS INDEXED
          </span>
        </footer>
      </main>
      {creating && (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center bg-stone-950/30 p-5"
          onClick={() => setCreating(false)}
        >
          <form
            role="dialog"
            aria-modal="true"
            aria-label="Create workspace table"
            onClick={(event) => event.stopPropagation()}
            onSubmit={(event) => {
              event.preventDefault()
              if (tableName.trim()) {
                setCustom([...custom, tableName.trim()])
                setTableName("")
                setCreating(false)
              }
            }}
            className="w-full max-w-md rounded-xl border border-border bg-background p-7 shadow-xl"
          >
            <div className="flex items-center justify-between">
              <h2 className="font-serif text-3xl">Create a workspace</h2>
              <button
                type="button"
                aria-label="Close"
                onClick={() => setCreating(false)}
              >
                <Icon name="close" />
              </button>
            </div>
            <p className="mt-3 text-xs leading-6 text-stone-500">
              Organize your contracts and build a structured audit table.
            </p>
            <label className="mt-6 block text-xs">
              Workspace name
              <input
                autoFocus
                required
                value={tableName}
                onChange={(event) => setTableName(event.target.value)}
                placeholder="e.g. Enterprise Vendor Audit"
                className="mt-2 w-full rounded-lg border border-border bg-white p-3 text-xs"
              />
            </label>
            <button className={`${primary} mt-5 w-full`}>
              Create Table
              <Icon name="plus" />
            </button>
          </form>
        </div>
      )}
    </div>
  )
}

function exportAudit(query: string, response: AuditResponse | null) {
  if (!response) return
  const lines = [
    "MEETHAQ AUDIT", "", "Question: " + query, "",
    response.answer, "", "Sources:",
    ...response.sources.flatMap((source) => [
      source.label + " " + source.source + " #" + source.chunk_index,
      "Chunk ID: " + (source.chunk_id || "unavailable"),
      source.text, "",
    ]),
  ]
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" }))
  const link = document.createElement("a")
  link.href = url
  link.download = "Meethaq-Audit-Evidence.txt"
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function CitedAnswer({
  answer = "",
  sources = [],
  onCite,
}: {
  answer?: string
  sources?: SourceItem[]
  onCite: (source: SourceItem) => void
}) {
  const safeAnswer = typeof answer === "string" ? answer : ""
  const safeSources = Array.isArray(sources) ? sources : []
  const parts = safeAnswer.split(/(\[Source [1-9]\d*\])/g)
  return (
    <p className="mt-4 whitespace-pre-wrap text-xs leading-7 text-stone-500">
      {parts.map((part, index) => {
        const match = part.match(/^\[Source ([1-9]\d*)\]$/)
        if (!match) {
          return <span key={`${part}-${index}`}>{part}</span>
        }
        const cited = sourceForCitation(safeSources, Number(match[1]))
        return (
          <button
            key={`${part}-${index}`}
            type="button"
            aria-controls="clause-inspector"
            disabled={!cited}
            onClick={() => cited && onCite(cited)}
            className="mx-0.5 inline-flex cursor-pointer items-center rounded-full border border-border bg-muted px-2 py-0.5 font-mono text-[8px] text-stone-700 hover:border-stone-500 hover:bg-stone-200"
          >
            {part}
          </button>
        )
      })}
    </p>
  )
}

function AuditWorkspace() {
  const [params] = useSearchParams()
  const initialQuery = params.get("q") || ""
  const {
    searchQuery,
    auditResponse,
    selectedSource,
    setSelectedSource,
    isLoading,
    errorMessage,
    runAudit,
  } = useMeethaq()
  const [query, setQuery] = useState("")
  const [notice, setNotice] = useState("")
  const displayedQuery = searchQuery || initialQuery || "Ask a question about the indexed contracts."
  const sources = auditResponse?.sources ?? []
  const lastRequested = useRef("")

  useEffect(() => {
    const nextQuery = params.get("q") || initialQuery
    if (!nextQuery.trim() || lastRequested.current === nextQuery) {
      return
    }
    if (searchQuery === nextQuery && (isLoading || auditResponse)) {
      lastRequested.current = nextQuery
      return
    }
    lastRequested.current = nextQuery
    void runAudit(nextQuery)
    // A follow-up changes provider state, but must not replay the URL question.
  }, [initialQuery, runAudit])

  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto max-w-[1400px] px-5 py-8 md:px-10">
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-[11px] text-stone-500"
        >
          <Icon name="left" className="size-3" />
          Command Center
        </Link>
        <div className="mb-7 mt-6 flex items-center justify-between gap-4">
          <div>
            <h1 className="font-serif text-4xl">Contract audit workspace</h1>
            <p className="mt-2 text-xs text-stone-500">
              Enterprise Vendor Audit · Side-by-side evidence review
            </p>
          </div>
          <button className={secondary} disabled={!auditResponse || isLoading} onClick={() => exportAudit(displayedQuery, auditResponse)}>
            <Icon name="download" />
            Export Evidence
          </button>
        </div>
        {errorMessage && (
          <p
            role="alert"
            className="mb-5 rounded-lg border border-[#e9d7c3] bg-[#f7f0e7] p-4 text-xs leading-7 text-[#8b6141]"
          >
            {errorMessage}
          </p>
        )}
        <div className="grid overflow-hidden rounded-xl border border-border bg-white lg:grid-cols-[1.2fr_1fr]">
          <section className="flex min-h-[620px] flex-col p-6 md:p-8">
            <div className="flex items-center gap-2 text-[10px] text-stone-500">
              <Icon name="chat" />
              AUDIT CONVERSATION
              <span className="ml-auto font-mono text-[8px]">
                {isLoading ? "RETRIEVING LOCAL CONTEXT" : "LOCAL REFERENCE SESSION"}
              </span>
            </div>
            <div className="ml-8 mt-7 rounded-lg bg-muted p-4 text-xs leading-7">
              {displayedQuery}
            </div>
            <div className="mt-8 flex items-center gap-2 text-xs font-medium">
              <Icon name="shield" />
              Meethaq AI
            </div>
            {isLoading ? (
              <>
                <h2 className="mt-5 font-serif text-[27px] leading-8">
                  Retrieving grounded clauses…
                </h2>
                <p className="mt-4 text-xs leading-7 text-stone-500">
                  Cosine retrieval and evidence checks are running against
                  the local contract index.
                </p>
                <div className="mt-5 h-[3px] w-full overflow-hidden rounded-full bg-[#e8e3db]">
                  <div className="h-full w-[45%] animate-pulse rounded-full bg-stone-600" />
                </div>
              </>
            ) : (
              <>
                <h2 className="mt-5 font-serif text-[27px] leading-8">
                  {auditResponse
                    ? (sources.length ? "Retrieved contract evidence." : "No supported answer found.")
                    : "Your audit is ready to be configured."}
                </h2>
                {auditResponse?.stats.calibration_status && auditResponse.stats.calibration_status !== "calibrated" && (
                  <p role="status" className="mt-3 text-[10px] leading-6 text-stone-500">Retrieval calibration is required for the current contract index.</p>
                )}
                {auditResponse ? (
                  <CitedAnswer
                    answer={auditResponse.answer}
                    sources={sources}
                    onCite={setSelectedSource}
                  />
                ) : (
                  <p className="mt-4 text-xs leading-7 text-stone-500">
                    Submit a compliance question from the command center to
                    retrieve source-grounded contract evidence.
                  </p>
                )}
              </>
            )}
            <div className="mt-4 flex flex-wrap gap-2">
              {sources.map((item) => (
                <button
                  key={`${item.source}-${item.chunk_index}-${item.label}`}
                  onClick={() => setSelectedSource(item)}
                  className={`flex items-center gap-1.5 rounded-full border px-3 py-2 font-mono text-[8px] ${
                    selectedSource?.label === item.label &&
                    selectedSource?.source === item.source &&
                    selectedSource?.chunk_index === item.chunk_index
                      ? "border-stone-500 bg-muted"
                      : "border-border"
                  }`}
                >
                  <Icon name="file" className="size-3" />
                  {item.label} · {item.source} · #{item.chunk_index}
                </button>
              ))}
            </div>
            <form
              className="mt-auto flex gap-2 pt-8"
              onSubmit={(event) => {
                event.preventDefault()
                if (query.trim()) {
                  lastRequested.current = query.trim()
                  void runAudit(query.trim())
                  setQuery("")
                }
              }}
            >
              <input
                aria-label="Follow-up question"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask a follow-up question..."
                className="min-w-0 flex-1 rounded-lg border border-border px-3 py-2 text-xs"
              />
              <button className={primary} disabled={isLoading}>
                Send
                <Icon name="arrow" className="size-3" />
              </button>
            </form>
          </section>
          <aside id="clause-inspector" className="border-t border-border bg-[#f7f4ef] p-6 md:p-8 lg:border-t-0 lg:border-l">
            <div className="flex items-center justify-between">
              <h2 className="font-serif text-[26px]">
                Evidence & clause inspector
              </h2>
              <span className="font-mono text-[9px] text-stone-500">
                {selectedSource != null && selectedSource.rerank_score != null
                  ? Number(selectedSource.rerank_score).toFixed(3)
                  : "—"}
              </span>
            </div>
            <p className="mt-2 text-[10px] text-stone-500">
              Cosine similarity ? Exact contract excerpts
            </p>
            {sources.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {sources.map((item) => {
                  const isSelected =
                    selectedSource?.label === item.label &&
                    selectedSource?.source === item.source &&
                    selectedSource?.chunk_index === item.chunk_index
                  return (
                    <button
                      key={`aside-${item.source}-${item.chunk_index}-${item.label}`}
                      onClick={() => setSelectedSource(item)}
                      className={`flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[9px] transition-colors ${
                        isSelected
                          ? "border-stone-600 bg-white font-medium text-stone-900 shadow-xs"
                          : "border-border bg-white/60 text-stone-600 hover:border-stone-400 hover:bg-white"
                      }`}
                    >
                      <Icon name="file" className="size-3" />
                      <span>{item.label}</span>
                      <span className="text-[8px] text-stone-400">
                        score{" "}
                        {item?.rerank_score != null
                          ? Number(item.rerank_score).toFixed(3)
                          : "—"}
                      </span>
                    </button>
                  )
                })}
              </div>
            )}
            <div className="mt-5 rounded-lg border border-border bg-white p-5">
              <div className="flex items-center gap-2 font-mono text-[10px]">
                <Icon name="file" />
                {selectedSource?.source || "No source selected"}
                <span className="ml-auto text-[8px] text-stone-400">
                  #{selectedSource?.chunk_index ?? "—"}
                </span>
              </div>
              <div className="mt-4 border-t border-border pt-4">
                <p className="font-mono text-[8px] tracking-wider text-stone-400">
                  {selectedSource?.label || "SOURCE"} - COSINE SIMILARITY{" "}
                  {selectedSource != null && selectedSource.rerank_score != null
                    ? Number(selectedSource.rerank_score).toFixed(3)
                    : "—"}
                </p>
                <p className="mt-4 font-serif text-lg leading-8">
                  {selectedSource?.text ? (
                    <mark className="bg-[#f0e4cf] px-1">{selectedSource.text}</mark>
                  ) : (
                    "Select a source chip or [Source N] citation to inspect the retrieved clause."
                  )}
                </p>
              </div>
            </div>
            <p className="mt-6 text-[10px] leading-6 text-stone-500">Review the exact retrieved clauses using their source citations.</p>
            <button
              className={`${secondary} mt-6 w-full`}
              onClick={async () => {
                try {
                  const proof = selectedSource
                    ? `${selectedSource.label || "[Source]"} ${selectedSource.source || "Document"} #${selectedSource.chunk_index ?? 0} (cosine similarity ${selectedSource.rerank_score != null ? Number(selectedSource.rerank_score).toFixed(3) : "0.000"}): ${selectedSource.text || auditResponse?.answer || ""}`
                    : auditResponse?.answer ||
                      "No live source selected. Run an audit to copy grounded proof."
                  await navigator.clipboard.writeText(proof)
                  setNotice("Legal proof copied.")
                } catch {
                  setNotice("Clipboard access unavailable.")
                }
              }}
            >
              <Icon name="copy" />
              Copy Legal Proof
            </button>
            {notice && (
              <p
                role="status"
                className="mt-3 text-center text-xs text-stone-600"
              >
                {notice}
              </p>
            )}
            <p className="mt-5 text-[10px] leading-6 text-stone-400">
              Demonstration audit. Original PDFs and live indexing require a
              connected local document repository. Legal review is recommended.
            </p>
          </aside>
        </div>
      </main>
    </div>
  )
}

const routerOptions =
  import.meta.env.BASE_URL && import.meta.env.BASE_URL !== "/"
    ? { basename: import.meta.env.BASE_URL }
    : undefined

const router = createBrowserRouter(
  [
    {
      path: "/",
      Component: Dashboard,
      ErrorBoundary: RouteErrorFallback,
    },
    {
      path: "/audit",
      Component: AuditWorkspace,
      ErrorBoundary: RouteErrorFallback,
    },
    {
      path: "*",
      Component: Dashboard,
      ErrorBoundary: RouteErrorFallback,
    },
  ],
  routerOptions,
)

export default function App() {
  return (
    <ErrorBoundary>
      <MeethaqProvider>
        <RouterProvider router={router} />
      </MeethaqProvider>
    </ErrorBoundary>
  )
}
