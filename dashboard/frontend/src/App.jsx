import { useState, useEffect, useCallback } from 'react'
import { 
  Activity, DollarSign, Zap, AlertTriangle, Server, 
  TrendingDown, Shield, Clock, ChevronRight, RefreshCw,
  ArrowUpRight, Play, Pause, SkipForward
} from 'lucide-react'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts'

// ─────────────────────────────── Mock Data ────────────────────────────────────
const INSTANCE_TYPES = ['c6i.2xlarge', 'g4dn.xlarge', 'r6i.xlarge', 'm6i.4xlarge', 'c6i.xlarge', 'g5.2xlarge']
const AZS = ['us-east-1a', 'us-east-1b', 'us-east-1c']

function generateMockJobs() {
  return Array.from({ length: 5 }, (_, i) => ({
    job_id: `job-${String(i + 1).padStart(3, '0')}`,
    status: ['running', 'running', 'migrating', 'running', 'paused'][i],
    instance_type: INSTANCE_TYPES[i % INSTANCE_TYPES.length],
    az: AZS[i % 3],
    progress: [0.67, 0.23, 0.51, 0.89, 0.41][i],
    cumulative_cost_usd: [8.34, 2.11, 5.88, 14.2, 4.07][i],
    budget_cap_usd: 50.0,
    n_migrations: [2, 0, 3, 1, 1][i],
    n_interruptions: [0, 0, 0, 0, 0][i],
    running_since: Date.now() / 1000 - [14400, 3600, 9000, 28800, 7200][i],
  }))
}

function generatePriceHistory() {
  const now = Date.now()
  return Array.from({ length: 24 }, (_, i) => ({
    time: new Date(now - (23 - i) * 3600 * 1000).toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' }),
    spot: +(0.24 + Math.sin(i * 0.5) * 0.06 + Math.random() * 0.02).toFixed(4),
    onDemand: 0.384,
  }))
}

function generateDecisionLog() {
  const actions = ['STAY', 'STAY', 'MIGRATE→c6i.xlarge', 'STAY', 'STAY', 'MIGRATE→r6i.xlarge', 'STAY', 'STAY', 'STAY', 'PAUSE']
  const confs = [0.91, 0.88, 0.73, 0.94, 0.92, 0.81, 0.95, 0.93, 0.90, 0.76]
  const reasons = [
    'Price stable (0.24/hr), risk low', 'Price stable, progress 22%',
    'Price spike +38% above baseline', 'New instance stable',
    'Progress 51%, cost on track', 'Interruption risk High → Medium AZ switch',
    'r6i stable, cost efficient', 'Approaching budget 60% checkpoint',
    'Progress 82%, staying course', 'Price spike — pausing to wait'
  ]
  return actions.map((action, i) => ({
    timestamp: Date.now() / 1000 - (actions.length - i) * 65,
    action,
    confidence: confs[i],
    reason: reasons[i],
    cost_impact: action.startsWith('STAY') ? -0.024 : action.startsWith('MIGRATE') ? -0.18 : -0.0,
  }))
}

// ─────────────────────────────── Components ───────────────────────────────────

function MetricCard({ icon: Icon, label, value, sub, color = 'default', glow = false }) {
  return (
    <div className="card" style={{ padding: '20px 24px', position: 'relative', overflow: 'hidden' }}>
      {glow && (
        <div style={{
          position: 'absolute', inset: 0,
          background: `radial-gradient(ellipse 60% 50% at 50% 0%, ${
            color === 'green' ? 'rgba(52,211,153,0.08)' :
            color === 'amber' ? 'rgba(251,191,36,0.08)' :
            'rgba(56,189,248,0.08)'
          }, transparent)`,
          pointerEvents: 'none'
        }} />
      )}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
            {label}
          </p>
          <div className={`metric-value ${color !== 'default' ? color : ''}`}>{value}</div>
          {sub && <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 6 }}>{sub}</p>}
        </div>
        <div style={{
          width: 40, height: 40, borderRadius: 'var(--radius-sm)',
          background: color === 'green' ? 'rgba(52,211,153,0.1)' :
                      color === 'amber' ? 'rgba(251,191,36,0.1)' :
                      'rgba(56,189,248,0.1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <Icon size={18} color={
            color === 'green' ? 'var(--accent-green)' :
            color === 'amber' ? 'var(--accent-amber)' :
            'var(--accent-electric)'
          } />
        </div>
      </div>
    </div>
  )
}

function JobRow({ job, onOverride }) {
  const statusBadge = job.status === 'running' ? 'badge-running' :
                      job.status === 'migrating' ? 'badge-migrating' :
                      job.status === 'paused' ? 'badge-paused' : 'badge-failed'
  const budgetPct = (job.cumulative_cost_usd / job.budget_cap_usd * 100).toFixed(0)
  const elapsed = Math.floor((Date.now() / 1000 - job.running_since) / 3600)

  return (
    <tr>
      <td>
        <span className="mono" style={{ color: 'var(--accent-electric)', fontSize: 12 }}>{job.job_id}</span>
      </td>
      <td><span className={`badge ${statusBadge}`}>{job.status}</span></td>
      <td>
        <span className="mono" style={{ fontSize: 12 }}>{job.instance_type}</span>
        <br />
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{job.az}</span>
      </td>
      <td style={{ minWidth: 120 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="progress-bar" style={{ flex: 1 }}>
            <div className="progress-fill" style={{ width: `${(job.progress * 100).toFixed(0)}%` }} />
          </div>
          <span style={{ fontSize: 12, minWidth: 32, color: 'var(--text-primary)', fontWeight: 600 }}>
            {(job.progress * 100).toFixed(0)}%
          </span>
        </div>
      </td>
      <td>
        <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>
          ${job.cumulative_cost_usd.toFixed(2)}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 11 }}> / ${job.budget_cap_usd}</span>
        <div style={{ fontSize: 10, color: budgetPct > 80 ? 'var(--accent-red)' : 'var(--text-muted)', marginTop: 2 }}>
          {budgetPct}% used
        </div>
      </td>
      <td style={{ textAlign: 'center' }}>
        <span style={{ color: job.n_migrations > 0 ? 'var(--accent-amber)' : 'var(--text-muted)' }}>
          {job.n_migrations}
        </span>
      </td>
      <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{elapsed}h ago</td>
      <td>
        <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: 11 }}
          onClick={() => onOverride(job)}>
          Override
        </button>
      </td>
    </tr>
  )
}

function DecisionLogPanel({ decisions }) {
  return (
    <div className="card" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
        <Zap size={16} color="var(--accent-electric)" />
        <h3 style={{ fontSize: 14, fontWeight: 700 }}>Agent Decision Log</h3>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)' }}>Live · last 60s</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {decisions.map((d, i) => {
          const isAction = d.action.startsWith('MIGRATE') || d.action.startsWith('PAUSE')
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'flex-start', gap: 12,
              padding: '10px 14px',
              background: isAction ? 'rgba(251,191,36,0.05)' : 'rgba(255,255,255,0.02)',
              border: `1px solid ${isAction ? 'rgba(251,191,36,0.15)' : 'transparent'}`,
              borderRadius: 'var(--radius-sm)',
              transition: 'all 0.2s'
            }}>
              {/* Confidence bars */}
              <div className="conf-bar-wrap" style={{ flexShrink: 0 }}>
                {[1,2,3,4,5].map(level => (
                  <div key={level} className={`conf-bar ${level <= Math.round(d.confidence * 5) ? 'active' : ''}`}
                    style={{ height: `${level * 4}px` }} />
                ))}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <span className={`mono`} style={{
                    fontSize: 12, fontWeight: 700,
                    color: d.action === 'STAY' ? 'var(--accent-green)' :
                           d.action.startsWith('MIGRATE') ? 'var(--accent-amber)' :
                           'var(--accent-purple)'
                  }}>
                    {d.action}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {(d.confidence * 100).toFixed(0)}% confidence
                  </span>
                  <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)' }}>
                    {new Date(d.timestamp * 1000).toLocaleTimeString()}
                  </span>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 3 }}>
                  {d.reason}
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function CostChart({ priceHistory }) {
  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{
        background: 'rgba(11,15,30,0.95)', border: '1px solid rgba(99,179,237,0.2)',
        borderRadius: 8, padding: '10px 14px', fontSize: 12
      }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</p>
        <p style={{ color: 'var(--accent-electric)' }}>Spot: ${payload[0]?.value}/hr</p>
        <p style={{ color: 'var(--text-muted)' }}>On-Demand: ${payload[1]?.value}/hr</p>
      </div>
    )
  }

  return (
    <div className="card" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
        <TrendingDown size={16} color="var(--accent-green)" />
        <h3 style={{ fontSize: 14, fontWeight: 700 }}>Spot Price History</h3>
        <span style={{ fontSize: 11, color: 'var(--text-secondary)', marginLeft: 'auto' }}>
          c6i.2xlarge · us-east-1
        </span>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={priceHistory} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="spotGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="time" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} />
          <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <Area type="monotone" dataKey="spot" stroke="#38bdf8" strokeWidth={2} fill="url(#spotGrad)" />
          <Line type="monotone" dataKey="onDemand" stroke="rgba(255,255,255,0.15)" strokeWidth={1} strokeDasharray="4 4" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

function RiskHeatmap() {
  const instances = ['c6i.xlarge', 'c6i.2xlarge', 'r6i.xlarge', 'g4dn.xlarge', 'm6i.4xlarge']
  const azs = ['us-east-1a', 'us-east-1b', 'us-east-1c']
  const riskLevels = [
    [0.08, 0.12, 0.07], [0.11, 0.09, 0.14], [0.06, 0.08, 0.05],
    [0.28, 0.32, 0.19], [0.15, 0.11, 0.18]
  ]

  const riskColor = (r) => {
    if (r < 0.1) return { bg: 'rgba(52,211,153,0.15)', text: '#34d399', label: 'Low' }
    if (r < 0.2) return { bg: 'rgba(251,191,36,0.15)', text: '#fbbf24', label: 'Med' }
    return { bg: 'rgba(248,113,113,0.15)', text: '#f87171', label: 'High' }
  }

  return (
    <div className="card" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
        <Shield size={16} color="var(--accent-purple)" />
        <h3 style={{ fontSize: 14, fontWeight: 700 }}>Interruption Risk Heatmap</h3>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 4 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', padding: '0 8px 8px' }}>Instance</th>
              {azs.map(az => (
                <th key={az} style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, padding: '0 4px 8px', textAlign: 'center' }}>
                  {az.split('-').pop()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {instances.map((inst, i) => (
              <tr key={inst}>
                <td className="mono" style={{ fontSize: 11, padding: '3px 8px', color: 'var(--text-secondary)' }}>{inst}</td>
                {riskLevels[i].map((r, j) => {
                  const { bg, text, label } = riskColor(r)
                  return (
                    <td key={j} style={{ padding: 3, textAlign: 'center' }}>
                      <div style={{
                        background: bg, color: text, borderRadius: 6,
                        padding: '5px 8px', fontSize: 10, fontWeight: 700,
                        minWidth: 40
                      }}>
                        {label}
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─────────────────────────────── Main App ─────────────────────────────────────

export default function App() {
  const [jobs, setJobs] = useState(generateMockJobs())
  const [decisions, setDecisions] = useState(generateDecisionLog())
  const [priceHistory] = useState(generatePriceHistory())
  const [activeTab, setActiveTab] = useState('overview')
  const [lastRefresh, setLastRefresh] = useState(new Date())
  const [overrideModal, setOverrideModal] = useState(null)
  const [totalSavings] = useState(34.21)

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      setJobs(prev => prev.map(j => ({
        ...j,
        progress: Math.min(1, j.progress + (j.status === 'running' ? 0.002 : 0)),
        cumulative_cost_usd: j.cumulative_cost_usd + (j.status !== 'paused' ? 0.004 : 0),
      })))
      setLastRefresh(new Date())
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const running = jobs.filter(j => j.status === 'running').length
  const migrating = jobs.filter(j => j.status === 'migrating').length
  const totalCost = jobs.reduce((s, j) => s + j.cumulative_cost_usd, 0)
  const totalMigrations = jobs.reduce((s, j) => s + j.n_migrations, 0)

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* ── Header ── */}
      <header style={{
        borderBottom: '1px solid var(--glass-border)',
        background: 'rgba(5,7,16,0.8)',
        backdropFilter: 'blur(20px)',
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <div style={{ maxWidth: 1400, margin: '0 auto', padding: '14px 28px', display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: 'linear-gradient(135deg, #38bdf8, #0ea5e9)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 16px rgba(56,189,248,0.4)'
            }}>
              <Zap size={16} color="#050710" fill="#050710" />
            </div>
            <div>
              <h1 style={{ fontSize: 15, fontWeight: 800, letterSpacing: '-0.02em' }}>Fleet Manager</h1>
              <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: -1 }}>RL-Powered Spot Intelligence</p>
            </div>
          </div>

          {/* Nav */}
          <nav style={{ display: 'flex', gap: 4, marginLeft: 32 }}>
            {[
              { id: 'overview', label: 'Overview', icon: Activity },
              { id: 'jobs', label: 'Jobs', icon: Server },
              { id: 'agent', label: 'Agent', icon: Zap },
            ].map(({ id, label, icon: Icon }) => (
              <button key={id} onClick={() => setActiveTab(id)} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 14px', borderRadius: 8,
                fontSize: 13, fontWeight: 500, cursor: 'pointer',
                border: 'none', fontFamily: 'var(--sans)',
                background: activeTab === id ? 'rgba(56,189,248,0.12)' : 'transparent',
                color: activeTab === id ? 'var(--accent-electric)' : 'var(--text-secondary)',
                transition: 'all 0.15s'
              }}>
                <Icon size={14} />
                {label}
              </button>
            ))}
          </nav>

          {/* Status bar */}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <span className="dot dot-green" />
              <span style={{ color: 'var(--text-secondary)' }}>RL Agent Active</span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Updated {lastRefresh.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main style={{ flex: 1, maxWidth: 1400, margin: '0 auto', padding: '28px', width: '100%' }}>

        {/* Metric Cards Row */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 16, marginBottom: 24
        }}>
          <MetricCard icon={Server} label="Active Jobs" value={running} sub={`${migrating} migrating`} color="default" glow />
          <MetricCard icon={DollarSign} label="Total Cost Today" value={`$${totalCost.toFixed(2)}`} sub="across all jobs" color="amber" />
          <MetricCard icon={TrendingDown} label="Estimated Savings" value={`$${totalSavings.toFixed(2)}`} sub="vs Always On-Demand" color="green" glow />
          <MetricCard icon={Activity} label="Migrations Today" value={totalMigrations} sub="0 interruptions" color="default" />
          <MetricCard icon={Shield} label="Interruptions" value="0" sub="Last 24 hours" color="green" glow />
        </div>

        {activeTab === 'overview' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <CostChart priceHistory={priceHistory} />
              <RiskHeatmap />
            </div>
            <DecisionLogPanel decisions={decisions} />
          </div>
        )}

        {activeTab === 'jobs' && (
          <div className="card" style={{ padding: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
              <Server size={16} color="var(--accent-electric)" />
              <h3 style={{ fontSize: 14, fontWeight: 700 }}>Managed Jobs</h3>
              <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 'auto' }}>
                {jobs.length} / 20 job slots used
              </span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Job ID</th><th>Status</th><th>Instance</th>
                    <th>Progress</th><th>Cost</th><th>Migrations</th>
                    <th>Running</th><th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map(job => (
                    <JobRow key={job.job_id} job={job} onOverride={setOverrideModal} />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'agent' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <div className="card" style={{ padding: 24 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
                <Zap size={16} color="var(--accent-electric)" />
                <h3 style={{ fontSize: 14, fontWeight: 700 }}>PPO Agent Statistics</h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {[
                  { label: 'STAY', pct: 72, color: 'var(--accent-green)' },
                  { label: 'MIGRATE', pct: 21, color: 'var(--accent-amber)' },
                  { label: 'PAUSE', pct: 7, color: 'var(--accent-purple)' },
                ].map(({ label, pct, color }) => (
                  <div key={label}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
                      <span className="mono" style={{ color }}>{label}</span>
                      <span style={{ color: 'var(--text-secondary)' }}>{pct}%</span>
                    </div>
                    <div className="progress-bar" style={{ height: 8 }}>
                      <div style={{
                        height: '100%', width: `${pct}%`, background: color,
                        borderRadius: 99, boxShadow: `0 0 8px ${color}`,
                        transition: 'width 0.5s'
                      }} />
                    </div>
                  </div>
                ))}
                <div className="divider" />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  {[
                    { k: 'Model', v: 'PPO + LSTM' },
                    { k: 'Avg Confidence', v: '84%' },
                    { k: 'Training Steps', v: '2.0M' },
                    { k: 'Policy Version', v: 'v1.2.0' },
                  ].map(({ k, v }) => (
                    <div key={k} style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 8, padding: '10px 14px' }}>
                      <p style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{k}</p>
                      <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginTop: 4 }}>{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <DecisionLogPanel decisions={decisions} />
          </div>
        )}
      </main>

      {/* ── Override Modal ── */}
      {overrideModal && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
          backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 1000,
        }} onClick={() => setOverrideModal(null)}>
          <div className="card" style={{ padding: 28, maxWidth: 400, width: '90%' }}
            onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
              <AlertTriangle size={18} color="var(--accent-amber)" />
              <h3 style={{ fontSize: 15, fontWeight: 700 }}>Human Override</h3>
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
              Force job <span className="mono" style={{ color: 'var(--accent-electric)' }}>{overrideModal.job_id}</span> to migrate to on-demand instance, bypassing the RL agent.
            </p>
            <div style={{ background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)', borderRadius: 8, padding: '10px 14px', marginBottom: 20 }}>
              <p style={{ fontSize: 12, color: 'var(--accent-red)' }}>
                ⚠️ This will override the agent's decision. Circuit breaker will be paused for this job.
              </p>
            </div>
            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn btn-danger" style={{ flex: 1 }}
                onClick={() => setOverrideModal(null)}>
                Force On-Demand Migration
              </button>
              <button className="btn btn-ghost" onClick={() => setOverrideModal(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
