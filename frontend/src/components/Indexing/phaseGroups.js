/**
 * Display names of the phase picker's group headers.
 *
 * The backend (`phase_metadata.compute_element_group`) names two groups in
 * German, "Reine Elemente" (a single element) and "Sonstiges" (no elements
 * found). Those strings are KEYS: `indexing_controller` sorts on them and the
 * listing sends them back as `groups`. They stay as they are; what the user
 * reads is chosen here. Any other group is a chemical system ("Al-Cu-Fe") and
 * is shown as it is.
 */
const GROUP_KEYS = {
  'Reine Elemente': 'phaseDropdown.groupPureElements',
  'Sonstiges': 'phaseDropdown.groupOther',
};

/** The text to show for the backend's group name `group`. */
export function groupLabel(group, t) {
  const key = GROUP_KEYS[group];
  return key ? t(key) : group;
}
