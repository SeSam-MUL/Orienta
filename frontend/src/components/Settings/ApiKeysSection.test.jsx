// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent, waitFor } from '@testing-library/react';

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#000', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
    green: '#0f0', red: '#f00', yellow: '#fd0', cyan: '#0ff',
  },
  alpha: () => '#000',
  spacing: { sm: 4, md: 8, lg: 16 },
  Button: ({ children, ...p }) => <button {...p}>{children}</button>,
  Input: ({ value, onChange, type, ...p }) =>
    <input type={type || 'text'} value={value} onChange={onChange} {...p} />,
  GroupBox: ({ title, children }) => <fieldset><legend>{title}</legend>{children}</fieldset>,
  Label: ({ children, ...p }) => <label {...p}>{children}</label>,
}));

const saveApiKeysMock = vi.fn();
const testApiKeyMock = vi.fn();
const getMock = vi.fn();

vi.mock('../../services/api', () => ({
  settingsApi: {
    get: () => getMock(),
    saveApiKeys: (keys) => saveApiKeysMock(keys),
    testApiKey: (name) => testApiKeyMock(name),
  },
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

import ApiKeysSection from './ApiKeysSection';


describe('ApiKeysSection', () => {
  it('renders provider row + shows "not configured" when no key set', async () => {
    getMock.mockResolvedValue({ data: {
      api_keys: { materials_project: { configured: false, preview: '' } },
      config_path: '/home/user/.config/Kikuchipy/config.json',
    }});
    const { getByText, findByText } = render(<ApiKeysSection />);
    expect(getByText('Materials Project')).toBeTruthy();
    await findByText(/not configured/i);
  });

  it('shows masked preview when key IS configured', async () => {
    getMock.mockResolvedValue({ data: {
      api_keys: { materials_project: { configured: true, preview: '************abcd' } },
      config_path: '/x',
    }});
    const { findByText } = render(<ApiKeysSection />);
    const status = await findByText(/configured/i);
    expect(status.textContent).toMatch(/\*+abcd/);
  });

  it('Save button is disabled while input is empty', async () => {
    getMock.mockResolvedValue({ data: {
      api_keys: { materials_project: { configured: false, preview: '' } },
      config_path: '/x',
    }});
    const { getByText, findByPlaceholderText } = render(<ApiKeysSection />);
    await findByPlaceholderText(/mp_/);
    const saveBtn = getByText('Save').closest('button');
    expect(saveBtn.disabled).toBe(true);
  });

  it('typing a key + clicking Save calls saveApiKeys with the new value', async () => {
    getMock.mockResolvedValue({ data: {
      api_keys: { materials_project: { configured: false, preview: '' } },
      config_path: '/x',
    }});
    saveApiKeysMock.mockResolvedValue({ data: { saved: { materials_project: '****' } } });
    const { findByPlaceholderText, getByText } = render(<ApiKeysSection />);
    const input = await findByPlaceholderText(/mp_/);
    fireEvent.change(input, { target: { value: 'mp_brand_new_key' } });
    const saveBtn = getByText('Save').closest('button');
    expect(saveBtn.disabled).toBe(false);
    fireEvent.click(saveBtn);
    await waitFor(() =>
      expect(saveApiKeysMock).toHaveBeenCalledWith({ materials_project: 'mp_brand_new_key' })
    );
  });

  it('Test button is disabled when no key is configured', async () => {
    getMock.mockResolvedValue({ data: {
      api_keys: { materials_project: { configured: false, preview: '' } },
      config_path: '/x',
    }});
    const { getByText, findByPlaceholderText } = render(<ApiKeysSection />);
    await findByPlaceholderText(/mp_/);
    const testBtn = getByText('Test').closest('button');
    expect(testBtn.disabled).toBe(true);
  });

  it('Test button calls testApiKey + shows success feedback', async () => {
    getMock.mockResolvedValue({ data: {
      api_keys: { materials_project: { configured: true, preview: '****abcd' } },
      config_path: '/x',
    }});
    testApiKeyMock.mockResolvedValue({ data: { success: true, message: 'OK' } });
    const { getByText, findByText } = render(<ApiKeysSection />);
    await findByText(/configured/i);
    const testBtn = getByText('Test').closest('button');
    fireEvent.click(testBtn);
    await findByText(/OK/);
    expect(testApiKeyMock).toHaveBeenCalledWith('materials_project');
  });

  it('Show/Hide button toggles input type between password and text', async () => {
    getMock.mockResolvedValue({ data: {
      api_keys: { materials_project: { configured: false, preview: '' } },
      config_path: '/x',
    }});
    const { findByPlaceholderText, getByText } = render(<ApiKeysSection />);
    const input = await findByPlaceholderText(/mp_/);
    expect(input.type).toBe('password');
    fireEvent.click(getByText('Show'));
    expect(input.type).toBe('text');
    fireEvent.click(getByText('Hide'));
    expect(input.type).toBe('password');
  });
});
