# Hopper Data Studio - Design System

**Version**: 1.0
**Last Updated**: 2026-01-16
**Purpose**: Documentation for the modern, shadcn-inspired design system

---

## Overview

The Hopper Data Studio UI has been redesigned with a modern, professional aesthetic inspired by shadcn/ui design principles. This design system provides:

- **Consistent visual language** across all pages
- **Professional typography** using the Inter font family
- **Zinc-based color palette** for subtle, sophisticated interfaces
- **Reusable components** for cards, badges, buttons, and more
- **Smooth animations** and transitions for better UX
- **Responsive design** that works on different screen sizes

---

## Design Philosophy

### Core Principles

1. **Minimalism** - Clean, uncluttered interfaces that let data take center stage
2. **Consistency** - Unified design language across all pages and components
3. **Accessibility** - Good contrast ratios, clear typography, and semantic HTML
4. **Performance** - Lightweight CSS with no external frameworks
5. **Professionalism** - Enterprise-grade appearance suitable for engineering applications

### Inspiration

This design system is inspired by **shadcn/ui**, a popular React component library known for:
- Zinc-based color palette (neutral, sophisticated)
- Clean typography with Inter font
- Subtle shadows and borders
- Modern card-based layouts
- Minimal, professional aesthetic

---

## File Structure

```
HDA/
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── pages/
│   ├── _shared_styles.py           # Core styling module (NEW)
│   ├── _shared_sidebar.py          # Enhanced sidebar (UPDATED)
│   ├── _shared_widgets.py          # Enhanced widgets (UPDATED)
│   ├── 1_Test_Explorer.py          # Example updated page (UPDATED)
│   └── [other pages...]
├── app.py                          # Landing page (REDESIGNED)
└── DESIGN_SYSTEM.md               # This file
```

---

## Color Palette

### Zinc Scale (Primary Neutral Palette)

```
zinc-50:  #fafafa  - Lightest backgrounds
zinc-100: #f4f4f5  - Card backgrounds, muted sections
zinc-200: #e4e4e7  - Borders
zinc-300: #d4d4d8  - Hover borders
zinc-400: #a1a1aa  - Disabled text
zinc-500: #71717a  - Muted text, labels
zinc-600: #52525b  - Secondary text
zinc-700: #3f3f46  - Tertiary text
zinc-800: #27272a  - Dark elements
zinc-900: #18181b  - Primary buttons, dark text
zinc-950: #09090b  - Darkest text
```

### Status Colors

```
Success: #16a34a (green-600) on #dcfce7 (green-100)
Warning: #ca8a04 (yellow-600) on #fef9c3 (yellow-100)
Error:   #dc2626 (red-600) on #fee2e2 (red-100)
Info:    #2563eb (blue-600) on #dbeafe (blue-100)
```

---

## Typography

### Font Family

**Primary**: `Inter` (Google Fonts)
**Fallbacks**: `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`

### Type Scale

```
h1: 2.25rem (36px) - font-weight: 700 - Landing page titles
h2: 1.875rem (30px) - font-weight: 600 - Section headers
h3: 1.5rem (24px) - font-weight: 600 - Subsection headers
h4: 1.25rem (20px) - font-weight: 600 - Card titles
p:  0.875rem (14px) - font-weight: 400 - Body text
```

### Text Utilities

- `.text-muted` - Muted gray text (#71717a)
- `.text-sm` - Small text (0.875rem)
- `.text-xs` - Extra small text (0.75rem)
- `.font-semibold` - Font weight 600
- `.font-bold` - Font weight 700

---

## Components

### 1. Cards

**Base Card** (`.card`)
```css
background: white
border: 1px solid zinc-200
border-radius: 0.5rem
padding: 1.5rem
box-shadow: subtle
```

**Elevated Card** (`.card-elevated`)
```css
background: white
border: none
border-radius: 0.5rem
padding: 1.5rem
box-shadow: large
```

**Muted Card** (`.card-muted`)
```css
background: zinc-100
border: 1px solid zinc-200
border-radius: 0.5rem
padding: 1.5rem
```

**Usage in Python:**
```python
st.markdown("""
<div class="card">
    <h4>Card Title</h4>
    <p>Card content goes here.</p>
</div>
""", unsafe_allow_html=True)
```

### 2. Badges

**Badge Variants:**
- `.badge-default` - Gray badge for general use
- `.badge-success` - Green badge for success states
- `.badge-warning` - Yellow badge for warnings
- `.badge-error` - Red badge for errors
- `.badge-info` - Blue badge for information

**Usage in Python:**
```python
from pages._shared_styles import render_status_badge

render_status_badge("P0", status="error")
render_status_badge("Active", status="success")
```

### 3. Buttons

**Primary Button** (default Streamlit button)
```css
background: zinc-900
color: white
border-radius: 0.5rem
padding: 0.5rem 1rem
font-weight: 500
transition: smooth hover
```

**Download Button** (Streamlit download button)
```css
background: zinc-100
color: zinc-950
border: 1px solid zinc-200
hover: zinc-200 background
```

### 4. Inputs & Forms

All form inputs have been styled with:
- 1px zinc-200 border
- Focus state with ring effect
- Smooth transitions
- Proper padding and spacing

This applies to:
- Text inputs
- Number inputs
- Text areas
- Select boxes

### 5. Feature Cards

**Usage:**
```python
from pages._shared_styles import render_feature_card

render_feature_card(
    title="Single Test Analysis",
    description="Analyze individual tests with full QC and uncertainty.",
    icon="🔬"
)
```

### 6. Metric Cards

**Usage:**
```python
from pages._shared_styles import render_metric_card

render_metric_card(
    title="Engineering Integrity",
    value="P0 Core",
    subtitle="Traceability • Uncertainty • QC"
)
```

### 7. Page Headers

**Usage:**
```python
from pages._shared_styles import render_page_header

render_page_header(
    title="Single Test Analysis",
    description="Analyze individual cold flow or hot fire tests",
    badge_text="P0",
    badge_type="error"
)
```

---

## Layout Utilities

### Spacing

**Margin utilities:**
```css
.mb-1, .mb-2, .mb-3, .mb-4, .mb-6, .mb-8  /* bottom */
.mt-1, .mt-2, .mt-3, .mt-4, .mt-6, .mt-8  /* top */
```

Values: 0.25rem, 0.5rem, 0.75rem, 1rem, 1.5rem, 2rem

### Shadows

```css
--shadow-sm: subtle shadow for cards
--shadow: default shadow
--shadow-md: medium shadow for hover states
--shadow-lg: large shadow for elevated cards
--shadow-xl: extra large shadow for modals
```

---

## Usage Guide

### For New Pages

1. **Import the styling module:**
```python
from pages._shared_styles import apply_custom_styles, render_page_header

st.set_page_config(page_title="Your Page", page_icon="🚀", layout="wide")
apply_custom_styles()
```

2. **Use the page header:**
```python
render_page_header(
    title="Your Page Title",
    description="Brief description of what this page does",
    badge_text="P1",  # Optional priority badge
    badge_type="warning"
)
```

3. **Use cards for content sections:**
```python
st.markdown("""
<div class="card">
    <h3>Section Title</h3>
    <p>Content goes here...</p>
</div>
""", unsafe_allow_html=True)
```

### For Existing Pages

1. **Add styling import at the top:**
```python
from pages._shared_styles import apply_custom_styles, render_page_header
apply_custom_styles()
```

2. **Replace title with page header:**
```python
# OLD
st.title("My Page")
st.markdown("Description")

# NEW
render_page_header(
    title="My Page",
    description="Description"
)
```

3. **Enhance widgets with custom HTML:**
Use the card classes and styling utilities to enhance visual presentation.

---

## Streamlit Theme Configuration

Located in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#18181b"        # zinc-900
backgroundColor = "#ffffff"     # white
secondaryBackgroundColor = "#f4f4f5"  # zinc-100
textColor = "#09090b"          # zinc-950
font = "sans serif"
base = "light"
```

---

## Best Practices

### DO ✓

- Use the provided component functions (`render_page_header`, `render_feature_card`, etc.)
- Maintain consistent spacing using utility classes
- Use semantic HTML in markdown strings
- Test on different screen sizes
- Use the zinc palette for consistency

### DON'T ✗

- Don't use inline styles when utility classes exist
- Don't use bright, saturated colors (keep it neutral)
- Don't override Streamlit's core functionality unnecessarily
- Don't use custom fonts other than Inter
- Don't create one-off styles without adding them to `_shared_styles.py`

---

## Animations

### Fade In

Applied to most cards and sections:
```css
.animate-fade-in {
    animation: fadeIn 0.3s ease-out;
}
```

Creates smooth entry animations for better perceived performance.

---

## Accessibility

### Color Contrast

All text/background combinations meet WCAG AA standards:
- Body text: zinc-500 on white (contrast ratio > 4.5:1)
- Headers: zinc-950 on white (contrast ratio > 12:1)
- Buttons: white on zinc-900 (contrast ratio > 12:1)

### Focus States

All interactive elements have visible focus states with ring effects.

### Semantic HTML

Use proper heading hierarchy and semantic elements where possible.

---

## Migration Guide

### For Engineering Team

All core functionality remains unchanged - only the visual presentation has been improved.

**What Changed:**
- New color palette (zinc-based instead of purple gradients)
- Modern typography (Inter font)
- Enhanced components (cards, badges, buttons)
- Improved spacing and layout
- Professional animations

**What Stayed the Same:**
- All data processing logic
- All analysis functions
- Database structure
- File organization
- P0/P1/P2 architecture

### Updating Other Pages

To apply the new design to additional pages:

1. Add `from pages._shared_styles import apply_custom_styles` at the top
2. Call `apply_custom_styles()` after `st.set_page_config()`
3. Replace `st.title()` with `render_page_header()`
4. Wrap content sections in card divs
5. Use status badges for priority/status indicators

**Example:**
```python
import streamlit as st
from pages._shared_styles import apply_custom_styles, render_page_header

st.set_page_config(page_title="Analysis", page_icon="📊", layout="wide")
apply_custom_styles()

render_page_header(
    title="Single Test Analysis",
    description="Analyze individual tests with full integrity checks"
)

# Rest of page content...
```

---

## Future Enhancements

### Potential Additions

1. **Dark Mode** - Add dark theme variant
2. **Chart Styling** - Unified Plotly/Matplotlib theme
3. **Data Table Styling** - Enhanced DataFrame presentation
4. **Loading States** - Skeleton screens and better spinners
5. **Toast Notifications** - Non-blocking success/error messages
6. **Modal Dialogs** - Overlay dialogs for confirmations

### Customization

To customize the design system:

1. **Colors**: Modify the `COLORS` dict in `_shared_styles.py`
2. **Typography**: Update font-size values in the CSS
3. **Spacing**: Adjust spacing utilities
4. **Components**: Add new component functions to `_shared_styles.py`

---

## Support

For questions or issues with the design system:
- Review this documentation
- Check `_shared_styles.py` for all available components
- See `app.py` for implementation examples
- Refer to updated pages (Test Explorer) for usage patterns

---

**Designed and implemented**: 2026-01-16
**Design inspiration**: shadcn/ui, modern web design principles
**Technology**: Streamlit + Custom CSS + Python
