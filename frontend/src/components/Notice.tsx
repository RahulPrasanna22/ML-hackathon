import type { ReactNode } from 'react';

interface NoticeProps {
  tone: 'error' | 'warning' | 'info';
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}

export function Notice({ tone, title, children, action }: NoticeProps) {
  return (
    <div className={`notice notice--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <div className="notice__body">
        <p className="notice__title">{title}</p>
        {children && <div className="notice__text">{children}</div>}
      </div>
      {action && <div className="notice__action">{action}</div>}
    </div>
  );
}
