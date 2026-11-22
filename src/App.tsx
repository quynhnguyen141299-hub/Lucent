/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from 'react';
import { AnalyticsView } from './components/AnalyticsView';
import { DocumentViewerDrawer } from './components/DocumentViewerDrawer';
import { EvaluationView } from './components/EvaluationView';
import { Layout, type NavSection } from './components/Layout';
import { LibraryView } from './components/LibraryView';
import { PipelineView } from './components/PipelineView';
import { SearchView } from './components/SearchView';
import { SettingsView } from './components/SettingsView';
import { lucentApi } from './services/api';
import type { EngineSettings } from './types/lucent';

export default function App() {
  const [activeSection, setActiveSection] = useState<NavSection>('search');
  const [settings, setSettings] = useState<EngineSettings | null>(null);
  const [inspectedDoc, setInspectedDoc] = useState<{
    documentId: string;
    chunkId?: string;
  } | null>(null);

  useEffect(() => {
    lucentApi
      .getConfig()
      .then((res) => setSettings(res.settings))
      .catch(() => {});
  }, []);

  const handleInspectDocument = (documentId: string, chunkId?: string) => {
    setInspectedDoc({ documentId, chunkId });
  };

  return (
    <Layout activeSection={activeSection} onNavigate={setActiveSection}>
      {activeSection === 'search' && (
        <SearchView settings={settings} onInspectDocument={handleInspectDocument} />
      )}
      {activeSection === 'library' && (
        <LibraryView onInspectDocument={handleInspectDocument} />
      )}
      {activeSection === 'pipeline' && <PipelineView />}
      {activeSection === 'evaluation' && <EvaluationView />}
      {activeSection === 'analytics' && <AnalyticsView />}
      {activeSection === 'settings' && (
        <SettingsView onSettingsUpdated={(updated) => setSettings(updated)} />
      )}

      <DocumentViewerDrawer
        documentId={inspectedDoc?.documentId || null}
        highlightChunkId={inspectedDoc?.chunkId || null}
        onClose={() => setInspectedDoc(null)}
      />
    </Layout>
  );
}
