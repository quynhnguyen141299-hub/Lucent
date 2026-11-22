import React, { useState } from 'react';
import {
  Activity,
  BarChart3,
  BookOpen,
  FlaskConical,
  GitBranch,
  Menu,
  Search,
  Settings,
  X,
} from 'lucide-react';

export type NavSection =
  | 'search'
  | 'library'
  | 'pipeline'
  | 'evaluation'
  | 'analytics'
  | 'settings';

interface NavItem {
  id: NavSection;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'search', label: 'Search', icon: Search },
  { id: 'library', label: 'Document Library', icon: BookOpen },
  { id: 'pipeline', label: 'Retrieval Pipeline', icon: GitBranch },
  { id: 'evaluation', label: 'Model Evaluation', icon: FlaskConical },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'settings', label: 'Settings', icon: Settings },
];

const SECTION_TITLES: Record<NavSection, string> = {
  search: 'Search',
  library: 'Document Library',
  pipeline: 'Retrieval Pipeline',
  evaluation: 'Model Evaluation',
  analytics: 'Analytics',
  settings: 'Settings',
};

interface LayoutProps {
  activeSection: NavSection;
  onNavigate: (section: NavSection) => void;
  topAction?: React.ReactNode;
  children: React.ReactNode;
}

function LucentMark() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className="shrink-0"
    >
      <path
        d="M12 2.5L21 12L12 21.5L3 12L12 2.5Z"
        stroke="#3B82F6"
        strokeWidth="1.75"
      />
      <path
        d="M12 6.5L17 12L12 17.5L7 12L12 6.5Z"
        stroke="#8B5CF6"
        strokeWidth="1.5"
      />
      <circle cx="12" cy="12" r="1.75" fill="#E6EDF7" />
    </svg>
  );
}

export function Layout({ activeSection, onNavigate, topAction, children }: LayoutProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleNavClick = (id: NavSection) => {
    onNavigate(id);
    setMobileOpen(false);
  };

  return (
    <div className="min-h-screen bg-[#070B14] text-[#E6EDF7] flex flex-col lg:flex-row">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:flex-col lg:w-60 lg:shrink-0 border-r border-[#1E293B] bg-[#0A0F1C] sticky top-0 h-screen">
        <div className="h-14 px-5 flex items-center gap-2.5 border-b border-[#1E293B]">
          <LucentMark />
          <span className="font-display text-base font-semibold tracking-tight text-white whitespace-nowrap">
            Lucent
          </span>
        </div>

        <nav aria-label="Primary navigation" className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeSection === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleNavClick(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
                  active
                    ? 'bg-[#131B2E] text-white border border-[#2B3B59]'
                    : 'text-[#94A3B8] hover:text-[#E6EDF7] hover:bg-[#0D1321] border border-transparent'
                }`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${active ? 'text-[#3B82F6]' : 'text-[#64748B]'}`}
                />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="px-5 py-4 border-t border-[#1E293B] text-xs text-[#64748B] leading-relaxed">
          <p className="font-medium text-[#94A3B8]">Semantic Search &amp; Ranking</p>
          <p className="mt-0.5">Find what matters. Understand why.</p>
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/70"
            onClick={() => setMobileOpen(false)}
            aria-hidden="true"
          />
          <aside className="relative w-64 max-w-[80vw] bg-[#0A0F1C] border-r border-[#1E293B] flex flex-col h-full z-10">
            <div className="h-14 px-4 flex items-center justify-between border-b border-[#1E293B]">
              <div className="flex items-center gap-2.5">
                <LucentMark />
                <span className="font-display text-base font-semibold text-white">Lucent</span>
              </div>
              <button
                type="button"
                onClick={() => setMobileOpen(false)}
                aria-label="Close navigation menu"
                className="p-2 text-[#94A3B8] hover:text-white rounded-md"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <nav aria-label="Mobile navigation" className="flex-1 p-3 space-y-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = activeSection === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleNavClick(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-3 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
                      active
                        ? 'bg-[#131B2E] text-white border border-[#2B3B59]'
                        : 'text-[#94A3B8] hover:text-white hover:bg-[#0D1321] border border-transparent'
                    }`}
                  >
                    <Icon
                      className={`w-4 h-4 shrink-0 ${active ? 'text-[#3B82F6]' : 'text-[#64748B]'}`}
                    />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      )}

      {/* Main Content Shell */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 px-4 sm:px-6 lg:px-8 border-b border-[#1E293B] bg-[#070B14]/95 backdrop-blur-sm sticky top-0 z-30 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation menu"
              className="lg:hidden p-2 -ml-2 text-[#94A3B8] hover:text-white rounded-md"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2 text-sm min-w-0">
              <span className="lg:hidden font-display font-semibold text-white">Lucent</span>
              <span className="hidden lg:inline text-[#64748B]">Lucent</span>
              <span className="text-[#334155]" aria-hidden="true">/</span>
              <span className="font-medium text-[#E6EDF7] truncate">
                {SECTION_TITLES[activeSection]}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {topAction || (
              <button
                type="button"
                onClick={() => onNavigate(activeSection === 'library' ? 'search' : 'library')}
                className="px-3.5 py-2 text-xs font-medium rounded-md bg-[#131B2E] hover:bg-[#1B2640] text-[#E6EDF7] border border-[#24324D] transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5"
              >
                {activeSection === 'library' ? (
                  <>
                    <Search className="w-3.5 h-3.5 text-[#3B82F6]" />
                    <span>Open Search</span>
                  </>
                ) : (
                  <>
                    <Activity className="w-3.5 h-3.5 text-[#3B82F6]" />
                    <span>Manage Corpus</span>
                  </>
                )}
              </button>
            )}
          </div>
        </header>

        <main className="flex-1 w-full max-w-[1360px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
      </div>
    </div>
  );
}
