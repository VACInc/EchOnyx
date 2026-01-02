# UI optimization mockups (no code changes)

This document describes the visual and interaction changes shown in the provided mockups. It is a proposal only and does **not** modify any code.

## Global layout and styling
- Warmer neutral background with softer card borders to reduce visual noise.
- Higher-contrast headings for quicker scanning while keeping body text subdued.
- Consistent card radius and padding across dashboard, lists, and form blocks.
- Sidebar remains dark for strong navigation anchoring; active item gains a deeper fill.
- Support both a light and a dark mode using the same layout and component structure.
- Remove all Batch Jobs UI and functionality; uploads process sequentially in a single-item queue.
- Users can still upload multiple videos at once; they will be queued and processed in order.

## Dashboard updates
- **Top row metrics**: quick stats converted into uniform tiles with clearer numeric hierarchy.
- **Primary action**: “Upload Video” promoted to a top-right pill for faster access.
- **Model status**: condensed into pill rows for a quicker at-a-glance read.
- **Recent videos**: list items get subtle row backgrounds; statuses shown as small pills or progress meter.

## Videos list updates
- Search bar enlarged and simplified.
- Table header row added to clarify columns (Status/Duration/Size/Added).
- Status pills provide an immediate scan path.
- Pagination made more explicit with a filled “Next” call-to-action.

## Upload page updates
- Replace the full Upload page with a modal popup triggered by the “Upload” action/button.
- The popup contains the dropzone and any upload instructions; the dedicated upload route/page is removed.
- Pipeline steps move into a side panel for readability (or a secondary section within the popup if space is limited).
- Tips consolidated into a single highlighted card.

## Interaction: hardware hover detail
- **Goal**: hovering the **Hardware** metric on the dashboard should reveal the same hardware details shown in **Settings → Hardware**.
- **Behavior**:
  - On hover, display a popover/tooltip (or lightweight panel) anchored to the Hardware tile.
  - The content should mirror the Settings hardware block (GPU/CPU/RAM/storage, etc.), no new fields.
  - If hardware data is still detecting, show the same placeholder states as Settings.

## Source mockups
- `ui-mockups/dashboard-optimized.png`
- `ui-mockups/videos-optimized.png`
- `ui-mockups/upload-optimized.png`
