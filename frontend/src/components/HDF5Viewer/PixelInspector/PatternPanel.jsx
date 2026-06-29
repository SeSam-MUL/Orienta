import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { h5Api } from '../../../services/api';
import PatternCanvas from '../canvases/PatternCanvas';
import { colors, Button, Label } from '../../../theme/components';

function SliderRow({ label, min, max, value, onChange, format = (v) => v, step = 1, title }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }} title={title}>
      <span style={{ fontSize: '9pt', color: colors.textSecondary, width: 60 }}>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ flex: 1, accentColor: colors.purple, height: 3 }}/>
      <span style={{ fontSize: '9pt', color: colors.cyan, width: 40, textAlign: 'right',
        fontFamily: 'monospace' }}>{format(value)}</span>
    </div>
  );
}

export default function PatternPanel({
  isFileOpen, currentIndex,
  currentPattern, patternLoading, patternShape,
  enh, setEnh, handleEnhReset, handleEnhAuto,
}) {
  const { t } = useTranslation('hdf5viewer');
  const setEnhField = (field) => (val) => setEnh((prev) => ({ ...prev, [field]: val }));
  const [source, setSource] = useState('processed'); // 'processed' | 'raw' | 'background'
  const [rawImage, setRawImage] = useState(null);
  const [bgImage, setBgImage] = useState(null);

  // Fetch raw on demand whenever source switches to 'raw' or pixel changes
  useEffect(() => {
    if (source !== 'raw' || !isFileOpen || currentIndex == null) return;
    let cancelled = false;
    h5Api.getPattern(currentIndex, 'raw')
      .then((res) => { if (!cancelled) setRawImage(res.data?.image ?? null); })
      .catch((err) => {
        if (cancelled) return;
        console.warn('[PatternPanel] raw fetch failed', err);
        setRawImage(null);
      });
    return () => { cancelled = true; };
  }, [source, currentIndex, isFileOpen]);

  // Reset raw cache on pixel change so we don't show stale raw while new fetch
  // is in flight. Background is constant per file, no reset needed there.
  useEffect(() => {
    setRawImage(null);
  }, [currentIndex]);

  // Fetch background once when source switches to 'background' (cached)
  useEffect(() => {
    if (source !== 'background' || !isFileOpen || bgImage) return;
    let cancelled = false;
    h5Api.getBackground('processed')
      .then((res) => { if (!cancelled) setBgImage(res.data?.image ?? null); })
      .catch((err) => {
        if (cancelled) return;
        console.warn('[PatternPanel] bg fetch failed', err);
        setBgImage(null);
      });
    return () => { cancelled = true; };
  }, [source, isFileOpen, bgImage]);

  // Reset bg cache on file change (isFileOpen flips)
  useEffect(() => {
    if (!isFileOpen) setBgImage(null);
  }, [isFileOpen]);

  const displayImage = source === 'processed' ? currentPattern
                     : source === 'raw' ? rawImage
                     : bgImage;

  return (
    <div>
      {/* Source toggle */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 6, fontSize: '9pt', color: colors.text }}>
        {['processed', 'raw', 'background'].map((s) => (
          <label key={s} title={t('pattern.sourceTooltip')} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3 }}>
            <input
              type="radio"
              name="pattern-source"
              checked={source === s}
              onChange={() => setSource(s)}
              style={{ accentColor: colors.purple }}
            />
            {t(`pattern.source${s.charAt(0).toUpperCase() + s.slice(1)}`)}
          </label>
        ))}
      </div>

      <div style={{
        background: '#0a0a0a', height: 200,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        position: 'relative', borderRadius: 3, marginBottom: 6,
      }}>
        {patternLoading && source === 'processed' && (
          <span style={{ color: colors.textSecondary }}>{t('pattern.loading')}</span>
        )}
        {displayImage && <PatternCanvas base64={displayImage} enhancement={enh} />}
        {!displayImage && !patternLoading && (
          <span style={{ color: colors.textSecondary, fontSize: '8pt' }}>
            {t('pattern.noSourcePattern', { source: t(`pattern.source${source.charAt(0).toUpperCase() + source.slice(1)}`) })}
          </span>
        )}
        {patternShape?.[0] > 0 && (
          <div style={{
            position: 'absolute', top: 4, right: 4, padding: '2px 6px',
            background: 'rgba(0,0,0,0.65)', fontSize: '8pt', color: colors.textSecondary,
          }}>{patternShape[1]}×{patternShape[0]}</div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <Label small secondary>{t('pattern.enhance')}</Label>
        <label style={{ fontSize: '9pt' }} title={t('pattern.invertTooltip')}>
          <input type="checkbox" checked={enh.invert}
            onChange={(e) => setEnhField('invert')(e.target.checked)} /> {t('pattern.invert')}
        </label>
        <label style={{ fontSize: '9pt' }} title={t('pattern.histEqTooltip')}>
          <input type="checkbox" checked={enh.clahe}
            onChange={(e) => setEnhField('clahe')(e.target.checked)} /> {t('pattern.histEq')}
        </label>
        <div style={{ flex: 1 }} />
        <Button small onClick={handleEnhReset} title={t('pattern.resetTooltip')}>{t('pattern.reset')}</Button>
        <Button small variant="primary" onClick={handleEnhAuto} disabled={!displayImage} title={t('pattern.autoTooltip')}>{t('pattern.auto')}</Button>
      </div>

      <SliderRow label={t('pattern.brightness')} min={-100} max={100} value={enh.brightness}
        onChange={setEnhField('brightness')} format={(v) => v >= 0 ? `+${v}` : `${v}`} title={t('pattern.sliderTooltip')} />
      <SliderRow label={t('pattern.contrast')} min={10} max={300} value={enh.contrast}
        onChange={setEnhField('contrast')} format={(v) => `${v}%`} title={t('pattern.sliderTooltip')} />
      <SliderRow label={t('pattern.gamma')} min={10} max={300} value={enh.gamma}
        onChange={setEnhField('gamma')} format={(v) => `${v}%`} title={t('pattern.sliderTooltip')} />
    </div>
  );
}
