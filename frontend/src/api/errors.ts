export type ApiErrorKind = 'config' | 'network' | 'timeout' | 'http' | 'validation' | 'aborted';

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly detail: string | null;
  readonly url: string | null;

  constructor(
    kind: ApiErrorKind,
    message: string,
    opts: { status?: number; detail?: string | null; url?: string | null } = {},
  ) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = opts.status ?? null;
    this.detail = opts.detail ?? null;
    this.url = opts.url ?? null;
  }
}

export class ValidationError extends Error {
  readonly path: string;
  constructor(path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = 'ValidationError';
    this.path = path;
  }
}

/** Short headline for an error, suitable for a notice title. */
export function errorTitle(err: unknown): string {
  if (err instanceof EvaluationFailedError) return 'Laptop 1 reported a failure';
  if (!(err instanceof ApiError)) return 'Unexpected error';
  switch (err.kind) {
    case 'config':
      return 'Backend not configured';
    case 'network':
      return 'Laptop 1 is unreachable';
    case 'timeout':
      return 'Request timed out';
    case 'http':
      if (err.status === 404) return 'Not found on Laptop 1';
      if (err.status !== null && err.status >= 500) return 'Laptop 1 reported a server error';
      return `Request rejected (HTTP ${err.status ?? '?'})`;
    case 'validation':
      return 'Unexpected response from Laptop 1';
    case 'aborted':
      return 'Request cancelled';
  }
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

/** The backend answered, but reported that the evaluation itself failed. */
export class EvaluationFailedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EvaluationFailedError';
  }
}
