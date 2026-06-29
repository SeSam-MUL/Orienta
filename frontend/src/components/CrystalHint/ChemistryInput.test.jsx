// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent } from '@testing-library/react';
import ChemistryInput from './ChemistryInput';

afterEach(() => cleanup());

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
  },
}));

describe('ChemistryInput', () => {
  it('renders all current elements as chips', () => {
    const { getByText } = render(
      <ChemistryInput elements={['Al', 'Si', 'Cu']}
        onAdd={vi.fn()} onRemove={vi.fn()} edsElements={[]} />
    );
    expect(getByText('Al')).toBeTruthy();
    expect(getByText('Si')).toBeTruthy();
    expect(getByText('Cu')).toBeTruthy();
  });

  it('calls onRemove when × button is clicked', () => {
    const onRemove = vi.fn();
    const { getByTitle } = render(
      <ChemistryInput elements={['Al', 'Si']}
        onAdd={vi.fn()} onRemove={onRemove} edsElements={[]} />
    );
    const removeBtn = getByTitle('Remove Al');
    fireEvent.click(removeBtn);
    expect(onRemove).toHaveBeenCalledWith('Al');
  });

  it('normalises element symbol case on submit', () => {
    const onAdd = vi.fn();
    const { container } = render(
      <ChemistryInput elements={[]} onAdd={onAdd} onRemove={vi.fn()} edsElements={[]} />
    );
    const input = container.querySelector('input');
    fireEvent.change(input, { target: { value: 'fe' } });
    fireEvent.submit(input.closest('form'));
    expect(onAdd).toHaveBeenCalledWith('Fe');
  });

  it('shows EDS auto-detection count when EDS elements available', () => {
    const { container } = render(
      <ChemistryInput elements={['Al']}
        onAdd={vi.fn()} onRemove={vi.fn()}
        edsElements={['Fe Kα1', 'Al Kα1', 'Si Kα1']} />
    );
    expect(container.textContent).toMatch(/EDS detected: 3/);
  });

  it('does not call onAdd for empty draft', () => {
    const onAdd = vi.fn();
    const { container } = render(
      <ChemistryInput elements={[]} onAdd={onAdd} onRemove={vi.fn()} edsElements={[]} />
    );
    const input = container.querySelector('input');
    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.submit(input.closest('form'));
    expect(onAdd).not.toHaveBeenCalled();
  });
});
