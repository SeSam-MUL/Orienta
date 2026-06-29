// frontend/src/components/PoleFigure/PoleFigureWindow.jsx
// Standalone shell rendered in the DETACHED window (main.jsx branches on
// ?view=polefigure). Fresh React/zustand context — reads everything from the
// backend over REST, stays live via useStateVersionPoll inside PoleFigureView.
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import PoleFigureView from './PoleFigureView';
import { colors } from '../../theme/tokens';

export default function PoleFigureWindow() {
  const { t } = useTranslation('polefigure');
  useEffect(() => { document.title = t('polefigure:windowTitle'); }, [t]);
  return (
    <div style={{ height: '100vh', width: '100vw', background: colors.bg }}>
      <PoleFigureView />
    </div>
  );
}
