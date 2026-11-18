import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { parseModelComparison } from '../api/validate';
import { healthOk, TEST_API, tripList } from '../test/fixtures';
import { json, mockFetch } from '../test/mockFetch';
import { ModelComparisonPanel, sortModels } from './ModelComparison';

vi.mock('react-leaflet', async () => (await import('../test/leafletMock')).leafletMock);

/** Test-only payload in the shape of GET /models/comparison (values are arbitrary test numbers). */
const row = (key: string, name: string, top1: number | null, f1: number | null, logloss: number | null, fit: number | null, infer: number | null, selected = false) => ({
  key, name, family: `${name} family`, cv_top1_accuracy: top1, cv_top1_accuracy_std: top1 === null ? null : 0.01,
  cv_macro_f1: f1, cv_macro_f1_std: null, cv_log_loss: logloss, cv_route_continuity: null, train_fold_top1_accuracy: null,
  fit_time_s_per_fold: null, fit_time_s_all_train: fit, inference_ms_per_observation: infer, selected,
});
const payload = {
  source: 'reports/experiments/model_comparison.csv',
  evaluation: 'TRAIN partition, 5-fold GroupKFold cross-validation grouped by trip (80 trips)',
  not_evaluator_results: true,
  selection_metric: 'cross-validated observation-level top-1 accuracy (GroupKFold by trip)',
  selected_model: 'mb',
  models: [
    row('ma', 'Model A', 0.8, 0.85, 0.3, 2, 0.01),
    row('mb', 'Model B', 0.9, 0.95, 0.07, 20, 0.007, true),
    row('mc', 'Model C', 0.88, null, 0.1, 100, null),
  ],
  baseline: row('base', 'Baseline', 0.5, null, null, null, null),
  tuned_final: { name: 'Model B (tuned, deployed)', cv_top1_accuracy: 0.91, cv_top1_accuracy_std: 0.015, cv_macro_f1: 0.95, cv_log_loss: 0.08, fit_time_s_all_train: 60, inference_ms_per_observation: 0.02 },
  notes: ['note one'],
};

describe('parseModelComparison', () => {
  it('maps the wire format and keeps missing values as null', () => {
    const d = parseModelComparison(payload);
    expect(d.models).toHaveLength(3);
    expect(d.models[1]).toMatchObject({ key: 'mb', cvTop1Accuracy: 0.9, selected: true });
    expect(d.models[2].cvMacroF1).toBeNull();
    expect(d.baseline?.cvTop1Accuracy).toBe(0.5);
    expect(d.tunedFinal?.cvTop1Accuracy).toBe(0.91);
  });

  it('rejects responses without models or protocol description', () => {
    expect(() => parseModelComparison({ evaluation: 'x' })).toThrow(/models/);
    expect(() => parseModelComparison({ models: [] })).toThrow(/evaluation/);
    expect(() => parseModelComparison({ evaluation: 'x', models: [{ key: 'a', cv_top1_accuracy: 'high' }] })).toThrow(/finite number/);
  });
});

describe('sortModels', () => {
  it('sorts numerically and keeps missing values last in both directions', () => {
    const d = parseModelComparison(payload);
    expect(sortModels(d.models, 'cvMacroF1', 'desc').map((m) => m.key)).toEqual(['mb', 'ma', 'mc']);
    expect(sortModels(d.models, 'cvMacroF1', 'asc').map((m) => m.key)).toEqual(['ma', 'mb', 'mc']);
  });
});

describe('ModelComparisonPanel', () => {
  const ready = () => ({ status: 'ready' as const, data: parseModelComparison(payload), error: null });

  it('labels results as TRAIN cross-validation and highlights the selected model', () => {
    render(<ModelComparisonPanel state={ready()} />);
    const panel = screen.getByTestId('model-comparison');
    expect(within(panel).getByText(/TRAIN cross-validation · not evaluator-test results/)).toBeInTheDocument();
    const selected = screen.getByTestId('model-row-mb');
    expect(selected).toHaveClass('models__row--selected');
    expect(within(selected).getByText('Selected · final model')).toBeInTheDocument();
    expect(within(screen.getByTestId('model-row-ma')).getByText('Not selected')).toBeInTheDocument();
    // default sort: CV top-1 accuracy, best first
    const order = screen.getAllByTestId(/^model-row-/).map((r) => r.getAttribute('data-testid'));
    expect(order).toEqual(['model-row-mb', 'model-row-mc', 'model-row-ma']);
    expect(screen.getByTestId('model-tuned-final')).toHaveTextContent('91.00%');
  });

  it('shows "not recorded" instead of inventing missing values', () => {
    render(<ModelComparisonPanel state={ready()} />);
    const c = screen.getByTestId('model-row-mc');
    expect(within(c).getAllByText('not recorded')).toHaveLength(2); // macro-F1 and inference time
  });

  it('sorts when a column header is clicked and toggles direction', async () => {
    const user = userEvent.setup();
    render(<ModelComparisonPanel state={ready()} />);
    const header = screen.getByRole('button', { name: /CV log loss/ });
    await user.click(header); // lower is better -> ascending first
    expect(screen.getAllByTestId(/^model-row-/).map((r) => r.getAttribute('data-testid'))).toEqual(['model-row-mb', 'model-row-mc', 'model-row-ma']);
    expect(header.closest('th')).toHaveAttribute('aria-sort', 'ascending');
    await user.click(header);
    expect(screen.getAllByTestId(/^model-row-/)[0]).toHaveAttribute('data-testid', 'model-row-ma');
    expect(header.closest('th')).toHaveAttribute('aria-sort', 'descending');
  });

  it('reports a load failure quietly, without an alert', () => {
    render(<ModelComparisonPanel state={{ status: 'error', data: null, error: new Error('HTTP 404') }} />);
    expect(screen.getByTestId('model-comparison-error')).toHaveTextContent('Model comparison unavailable');
    expect(screen.queryByRole('alert')).toBeNull();
  });
});

describe('dashboard integration', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_API_BASE_URL', TEST_API);
    vi.stubEnv('VITE_DEV_FALLBACK_TRIPS', '');
  });

  it('loads GET /models/comparison from Laptop 1 and renders the table', async () => {
    const f = mockFetch({
      'GET /health': () => json(healthOk),
      'GET /evaluation/trips': () => json(tripList),
      'GET /models/comparison': () => json(payload),
    });
    render(<App />);
    expect(await screen.findByTestId('model-row-mb')).toBeInTheDocument();
    expect(f.calls.some((c) => c.url === `${TEST_API}/models/comparison`)).toBe(true);
  });
});
