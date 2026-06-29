/**
 * TreeTab — HDF5 tree browser with lazy loading + attribute inspector.
 *
 * Props:
 *   isFileOpen — whether an HDF5 file is currently open
 *
 * Local state (kept inside): tree root, loading flags, error,
 * selected path, attribute payload. Uses h5Api directly for
 * lazy-fetching child nodes — that's an API client, not UI state,
 * so it stays in this tab.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { h5Api } from '../../../services/api';
import { colors, alpha } from '../../../theme/components';
import EmptyState from './EmptyState';

// Single tree node — renders groups (collapsible) and datasets (leaf)
function TreeNode({ node, depth, onSelectPath, selectedPath }) {
  const { t } = useTranslation('hdf5viewer');
  const [expanded, setExpanded] = useState(depth === 0);
  const [lazyChildren, setLazyChildren] = useState(null);
  const [lazyLoading, setLazyLoading] = useState(false);
  const unmountedRef = useRef(false);

  useEffect(() => {
    return () => { unmountedRef.current = true; };
  }, []);

  const isGroup = node.type === 'group';
  const hasChildren = isGroup && (node.children_count > 0);
  const childrenLoaded = node.children && node.children.length > 0;
  const needsLazyLoad = hasChildren && !childrenLoaded && lazyChildren === null;
  const displayChildren = lazyChildren ?? node.children ?? [];

  const isSelected = selectedPath === node.path;

  const handleToggle = () => {
    if (!isGroup) return;
    if (!expanded && needsLazyLoad) {
      setLazyLoading(true);
      // Fetch only the direct children of this node (not the full tree)
      (async () => {
        try {
          const res = await h5Api.getTreeNode(node.path);
          if (unmountedRef.current) return;
          setLazyChildren(res.data?.children ?? []);
        } catch (err) {
          if (unmountedRef.current) return;
          console.warn(`[TreeTab] getTreeNode(${node.path}) failed`, err);
          setLazyChildren([]);
        } finally {
          if (!unmountedRef.current) setLazyLoading(false);
        }
      })();
    }
    setExpanded((e) => !e);
  };

  const indentPx = depth * 14;
  const iconColor = isGroup ? colors.yellow : colors.cyan;
  // const icon = isGroup ? (expanded ? '▾' : '▸') : '■';

  return (
    <div>
      {/* Node row */}
      <div
        onClick={() => {
          if (isGroup) handleToggle();
          if (node.path) onSelectPath(node.path);
        }}
        title={node.path}
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 5,
          padding: '3px 6px',
          paddingLeft: 6 + indentPx,
          cursor: 'pointer',
          background: isSelected ? alpha(colors.purple, 16) : 'transparent',
          borderLeft: isSelected ? `2px solid ${colors.purple}` : '2px solid transparent',
          userSelect: 'none',
          transition: 'background 0.1s, border-left-color 0.1s',
        }}
        onMouseEnter={(e) => {
          if (!isSelected) e.currentTarget.style.background = alpha(colors.border, 19);
        }}
        onMouseLeave={(e) => {
          if (!isSelected) e.currentTarget.style.background = 'transparent';
        }}
      >
        {/* Expand icon (groups only) */}
        <span style={{
          fontSize: '8pt',
          color: iconColor,
          width: 10,
          flexShrink: 0,
          lineHeight: 1,
          display: 'inline-block',
          transition: 'transform 0.15s',
          transform: isGroup ? (expanded ? 'rotate(0deg)' : 'rotate(-90deg)') : 'none',
        }}>
          {isGroup ? '▾' : '■'}
        </span>

        {/* Name */}
        <span title={node.path || node.name} style={{
          fontSize: '9pt',
          color: isGroup ? colors.text : colors.cyan,
          fontFamily: "'Courier New', monospace",
          flex: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {node.name}
        </span>

        {/* Right-side annotation */}
        {isGroup && node.children_count > 0 && (
          <span style={{
            fontSize: '7pt', color: colors.textSecondary, flexShrink: 0,
            padding: '0 5px', borderRadius: 8,
            background: alpha(colors.border, 19),
          }}>
            {node.children_count}
          </span>
        )}
        {!isGroup && node.shape?.length > 0 && (
          <span style={{
            fontSize: '8pt',
            color: colors.textSecondary,
            flexShrink: 0,
            fontFamily: "'Courier New', monospace",
          }}>
            {node.shape.join('×')} {node.dtype}
          </span>
        )}
      </div>

      {/* Children (when expanded) */}
      {isGroup && expanded && (
        <div>
          {lazyLoading && (
            <div style={{ padding: '3px 6px 3px ' + (6 + indentPx + 14) + 'px', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {[0.7, 0.5, 0.6].map((w, i) => (
                <div key={i} className="skeleton-shimmer" style={{
                  height: 12, borderRadius: 2,
                  width: `${w * 100}%`,
                  background: alpha(colors.border, 19),
                }} />
              ))}
            </div>
          )}
          {!lazyLoading && displayChildren.map((child) => (
            <TreeNode
              key={child.path ?? child.name}
              node={child}
              depth={depth + 1}
              onSelectPath={onSelectPath}
              selectedPath={selectedPath}
            />
          ))}
          {!lazyLoading && displayChildren.length === 0 && childrenLoaded === false && lazyChildren !== null && (
            <div style={{
              padding: '3px 6px',
              paddingLeft: 6 + indentPx + 14,
              fontSize: '8pt',
              color: colors.textSecondary,
            }}>
              {t('tree.empty')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function TreeTab({ isFileOpen }) {
  const { t } = useTranslation('hdf5viewer');
  const [tree, setTree] = useState(null);
  const [treeLoading, setTreeLoading] = useState(false);
  const [treeError, setTreeError] = useState(null);
  const [selectedPath, setSelectedPath] = useState(null);
  const [attrs, setAttrs] = useState(null);
  const [attrsLoading, setAttrsLoading] = useState(false);
  const selectionSeq = useRef(0);

  // Load tree when file opens
  useEffect(() => {
    if (!isFileOpen) { setTree(null); setSelectedPath(null); setAttrs(null); return; }
    let cancelled = false;
    setTreeLoading(true);
    setTreeError(null);
    (async () => {
      try {
        const res = await h5Api.getTree(5);
        if (cancelled) return;
        setTree(res.data);
      } catch (err) {
        if (cancelled) return;
        console.warn('[TreeTab] getTree failed', err);
        setTreeError(t('tree.loadFailed'));
      } finally {
        if (!cancelled) setTreeLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [isFileOpen]);

  // Load attributes when path selected — guard against stale responses on
  // rapid selection changes by tracking a monotonic selection sequence.
  const handleSelectPath = useCallback(async (path) => {
    const seq = ++selectionSeq.current;
    setSelectedPath(path);
    setAttrs(null);
    setAttrsLoading(true);
    try {
      const res = await h5Api.getAttributes(path);
      if (seq !== selectionSeq.current) return; // stale
      setAttrs(res.data);
    } catch (err) {
      if (seq !== selectionSeq.current) return;
      console.warn(`[TreeTab] getAttributes(${path}) failed`, err);
      setAttrs({ error: t('tree.attrLoadFailed') });
    } finally {
      if (seq === selectionSeq.current) setAttrsLoading(false);
    }
  }, []);

  if (!isFileOpen) return <EmptyState icon={'☷'}>{t('tree.openToBrowse')}</EmptyState>;

  if (treeLoading) return (
    <div style={{ flex: 1, padding: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
      {[0.8, 0.6, 0.9, 0.5, 0.7, 0.4].map((w, i) => (
        <div key={i} className="skeleton-shimmer" style={{
          height: 14, borderRadius: 3,
          width: `${w * 100}%`,
          background: alpha(colors.border, 19),
        }} />
      ))}
    </div>
  );

  if (treeError) return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '18pt', opacity: 0.5, marginBottom: 6 }}>{'⚠'}</div>
        <span style={{ fontSize: '9pt', color: colors.red }}>{treeError}</span>
      </div>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Tree scroll area */}
      <div className="thin-scrollbar" style={{ flex: '1 1 0', overflowY: 'auto', overflowX: 'auto', minHeight: 0 }}>
        {tree?.tree?.map((node) => (
          <TreeNode
            key={node.path ?? node.name}
            node={node}
            depth={0}
            onSelectPath={handleSelectPath}
            selectedPath={selectedPath}
          />
        ))}
        {tree && (!tree.tree || tree.tree.length === 0) && (
          <div style={{ padding: 12, fontSize: '9pt', color: colors.textSecondary, textAlign: 'center' }}>
            <span style={{ opacity: 0.3, marginRight: 4 }}>{'☷'}</span>
            {t('tree.noItems')}
          </div>
        )}
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: colors.border, flexShrink: 0 }} />

      {/* Attribute inspector panel */}
      <div className="thin-scrollbar" style={{
        flexShrink: 0,
        maxHeight: 220,
        overflowY: 'auto',
        padding: '6px 8px',
        background: colors.bgTertiary,
      }}>
        {!selectedPath && (
          <div style={{ fontSize: '8pt', color: colors.textSecondary, textAlign: 'center', padding: '12px 0' }}>
            {t('tree.clickToInspect')}
          </div>
        )}

        {selectedPath && (
          <>
            {/* Selected path header */}
            <div style={{
              fontSize: '8pt',
              color: colors.purple,
              fontFamily: "'Courier New', monospace",
              marginBottom: 6,
              wordBreak: 'break-all',
            }}>
              {selectedPath}
            </div>

            {attrsLoading && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {[0.6, 0.8, 0.5].map((w, i) => (
                  <div key={i} className="skeleton-shimmer" style={{ height: 10, width: `${w * 100}%`, borderRadius: 2, background: alpha(colors.border, 19) }} />
                ))}
              </div>
            )}

            {!attrsLoading && attrs?.error && (
              <div style={{ fontSize: '8pt', color: colors.red }}>{attrs.error}</div>
            )}

            {!attrsLoading && attrs && !attrs.error && (
              <>
                {/* Shape / dtype for datasets */}
                {attrs.shape?.length > 0 && (
                  <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: '8pt', color: colors.textSecondary, width: 44, flexShrink: 0 }}>{t('tree.shape')}</span>
                    <span style={{ fontSize: '8pt', color: colors.orange, fontFamily: "'Courier New', monospace" }}>
                      {attrs.shape.join(' × ')}
                    </span>
                  </div>
                )}
                {attrs.dtype && (
                  <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: '8pt', color: colors.textSecondary, width: 44, flexShrink: 0 }}>{t('tree.dtype')}</span>
                    <span style={{ fontSize: '8pt', color: colors.cyan, fontFamily: "'Courier New', monospace" }}>
                      {attrs.dtype}
                    </span>
                  </div>
                )}
                {attrs.size != null && (
                  <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: '8pt', color: colors.textSecondary, width: 44, flexShrink: 0 }}>{t('tree.size')}</span>
                    <span style={{ fontSize: '8pt', color: colors.text, fontFamily: "'Courier New', monospace" }}>
                      {attrs.size.toLocaleString()}
                    </span>
                  </div>
                )}

                {/* HDF5 attributes */}
                {attrs.attributes && Object.keys(attrs.attributes).length > 0 ? (
                  <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 4 }}>
                    <tbody>
                      {Object.entries(attrs.attributes).map(([k, v]) => (
                        <tr key={k}>
                          <td style={{
                            padding: '2px 6px 2px 0',
                            fontSize: '8pt',
                            color: colors.orange,
                            verticalAlign: 'top',
                            width: '38%',
                            wordBreak: 'break-word',
                          }}>
                            {k}
                          </td>
                          <td style={{
                            padding: '2px 0',
                            fontSize: '8pt',
                            color: colors.text,
                            wordBreak: 'break-all',
                            fontFamily: "'Courier New', monospace",
                          }}>
                            {String(v).substring(0, 120)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div style={{ fontSize: '8pt', color: colors.textSecondary, marginTop: 4 }}>
                    {t('tree.noAttributes')}
                  </div>
                )}

                {/* Group children list */}
                {attrs.children && attrs.children.length > 0 && (
                  <div style={{ marginTop: 4 }}>
                    <div style={{ fontSize: '8pt', color: colors.textSecondary, marginBottom: 2 }}>
                      {t('tree.children', { count: attrs.children.length })}
                    </div>
                    <div style={{ fontSize: '8pt', color: colors.text, fontFamily: "'Courier New', monospace" }}>
                      {attrs.children.slice(0, 20).join(', ')}
                      {attrs.children.length > 20 && t('tree.childrenMore', { count: attrs.children.length - 20 })}
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
