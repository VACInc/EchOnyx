# EchOnyx Design System

This is the binding spec for the EchOnyx frontend. Every page and component uses these
tokens and primitives — no hardcoded palette classes (`slate-*`, `blue-*`, …) in page
code except through the primitives and semantic utilities defined here.

## Brand

- **Identity:** "echo" (audio) + "onyx" (deep black stone). Local-first, private, precise.
- **Logo:** existing assets in `frontend/public/` and `img/` are canonical; do not restyle.
- **Accent:** the blue→indigo→violet gradient from the logo, used *sparingly*: primary CTA,
  active-progress fills, brand moments. Everything else is calm neutral surfaces.
- **Voice:** plain, specific, no jargon in user-facing copy. Every failure message says what
  to do next.

## Tokens

Tokens live in `frontend/app/globals.css` as HSL CSS variables with light values on `:root`
and dark values on `.dark`, mapped in `tailwind.config.ts`. Semantic roles:

| Role | Usage |
|---|---|
| `background` / `foreground` | app canvas and default text |
| `card` / `card-foreground` | raised surfaces (panels, rows, dialogs) |
| `primary` / `primary-foreground` | primary buttons, active nav, links |
| `secondary`, `muted`, `accent` | secondary buttons, subdued text/fills, highlights |
| `destructive` | dangerous actions and error fills |
| `success`, `warning`, `info` | status colors (add these three: e.g. `--success: 152 60% 38%`, `--warning: 38 92% 44%`, `--info: 217 91% 55%`, with dark-mode variants and `-foreground` pairs) |
| `border`, `input`, `ring` | hairlines, form borders, focus rings |
| `--radius` | 0.75rem base; `rounded-lg/md/sm` derive from it. Pills use `rounded-full`. |

Type: Inter (existing). Scale: page title `text-2xl font-semibold`; section heading
`text-lg font-semibold`; body `text-sm`; metadata `text-xs text-muted-foreground`.
No letter-spacing tricks (`tracking-[0.2em]` uppercase labels are retired except tiny
overline labels, which use `text-[11px] font-medium uppercase tracking-wide text-muted-foreground`).

Shadows: `shadow-sm` on cards, `shadow-lg` on overlays only. Motion: 150–200ms ease-out for
hover/focus, 200–300ms for enter/exit; respect `prefers-reduced-motion`.

## Primitives (`frontend/components/ui/`)

All interactive primitives are keyboard-accessible with visible `:focus-visible` rings.

- `button.tsx` — variants: `primary` (gradient brand), `secondary`, `outline`, `ghost`,
  `destructive`; sizes `sm`/`md`; `loading` prop renders spinner + disables.
- `card.tsx` — surface wrapper (`bg-card border border-border rounded-lg shadow-sm`).
- `badge.tsx` — status pill; variants map to semantic colors.
- `status-badge.tsx` — domain-aware badge translating raw statuses (`queued`, `processing`,
  `completed`, `failed`, `uploaded`, `loaded`, `cached`, `uncached`, `downloading`,
  `online`, `offline`) into labeled, icon-bearing pills with a `title`/tooltip explanation.
- `dialog.tsx` — modal with focus trap, Escape/overlay close, ARIA labelling.
- `confirm-dialog.tsx` + `useConfirm()` — promise-based replacement for `window.confirm`;
  supports destructive styling and a details line.
- `toast.tsx` + `useToast()` — top-right stack, success/error/info, auto-dismiss with
  manual close; errors persist longer.
- `tooltip.tsx` — hover *and* focus triggered, `aria-describedby`, works on touch (tap).
- `input.tsx`, `select.tsx`, `field.tsx` — labeled form controls with description and
  inline error slots.
- `tabs.tsx` — accessible tabs (`role="tablist"`, arrow-key navigation).
- `progress.tsx` — determinate bar (brand gradient fill) with `aria-valuenow`.
- `skeleton.tsx` — shimmer placeholder blocks.
- `empty-state.tsx` — icon + headline + hint + optional action button.
- `error-state.tsx` — inline error panel: what failed, why (message), retry action.
- `spinner.tsx` — sized loading indicator.
- `tag-input.tsx` — label entry with suggestions: fetches existing labels, filters as you
  type, keyboard selection, comma/Enter to add, click-to-remove chips.

## Patterns

- **Every data view has three states.** Loading = skeletons shaped like the content (not
  bare "Loading..."), empty = `EmptyState` with a next step, error = `ErrorState` with retry.
- **Every mutation reports.** Success → toast (or visible inline state change); failure →
  toast with the server's `detail` message; nothing goes only to `console.error`.
- **No native dialogs.** `window.alert`/`confirm` are banned; use `useToast`/`useConfirm`.
- **No cryptic shorthand.** Statuses/models are labeled with full words; abbreviations get
  a tooltip and a legend where space is tight.
- **Progressive disclosure.** Advanced settings and runtime internals are grouped and
  collapsible, visible on demand, never removed.
- **Destructive actions** use `ConfirmDialog` with explicit consequence text.

## Layout & responsiveness

- App shell: sidebar (16rem) on `lg+`; below `lg` a top bar with the logo and a hamburger
  opening the same nav as a drawer (overlay, focus-trapped, Escape closes).
- The sidebar keeps its branded dark surface in both themes; main content follows the theme.
- Main content: `p-4 sm:p-6`, max width `max-w-7xl mx-auto` on wide screens.
- Tables/grids collapse to stacked cards below `md`.

## Theme

- Class-based dark mode (`darkMode: "class"`). A tiny inline script in `app/layout.tsx`
  `<head>` applies the stored/system theme before paint (no flash).
- Theme choice: light / dark / system, persisted in `localStorage` (`echonyx-theme`).

## Accessibility baseline

- Semantic landmarks (`nav`, `main`, `header`), labeled controls, `aria-live="polite"` for
  toasts, alt text on images, focus management in overlays, WCAG AA contrast in both themes.
