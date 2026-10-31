import { useTranslation } from 'react-i18next';
import { colors } from '../../theme/components';

/**
 * The coded warnings of a FINISHED simulation job (`job.warnings`, from the
 * backend job's finishing step). Today one code exists:
 *
 *   masterUnreadableByKikuchipy -- the job produced a master that carries the
 *   pattern array but that kikuchipy cannot load (EMsoft's GPU master program
 *   writes such files), so the phase library does not offer it for Dictionary
 *   indexing. Without this line the job ends "completed" and the user finds out
 *   when the master is missing from the picker.
 *
 * A code this build does not know is still shown (code + file) rather than
 * dropped: a warning the page cannot translate is better than none.
 */
export default function JobWarnings({ warnings }) {
  const { t } = useTranslation('simulation');
  if (!Array.isArray(warnings) || warnings.length === 0) return null;
  return (
    <div data-testid="job-warnings" style={{ marginTop: 2 }}>
      {warnings.map((w, i) => {
        const text = w?.code === 'masterUnreadableByKikuchipy'
          ? t('jobWarnings.masterUnreadableByKikuchipy', {
              file: w.file || '?',
              program: w.program || t('jobWarnings.unknownProgram'),
              interpolation: { escapeValue: false },
            })
          : t('jobWarnings.unknown', {
              code: w?.code || '?', file: w?.file || '',
              interpolation: { escapeValue: false },
            });
        return (
          <div key={`${w?.code}-${w?.file}-${i}`} data-code={w?.code}
               style={{ fontSize: 11, lineHeight: 1.4, color: colors.orange }}>
            {'⚠ '}{text}
          </div>
        );
      })}
    </div>
  );
}
