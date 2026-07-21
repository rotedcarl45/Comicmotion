import streamlit as st
import os
import sys

# Ensure the project root is on the Python path regardless of where
# the user launches streamlit from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from core import config, database, workspace, validator
from services import comic_ingestion, panel_detector, gemini_vision, story_manager, tts_service, motion_renderer, final_renderer

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
ui_config = config.get("ui")
st.set_page_config(
    page_title=ui_config["page_title"],
    page_icon=ui_config["page_icon"],
    layout=ui_config["layout"],
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0d0d0f; }

    /* ---- Header ---- */
    .cm-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .cm-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .cm-header p {
        color: #8892b0;
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
    }

    /* ---- Cards ---- */
    .cm-card {
        background: #13131a;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ---- Status badges ---- */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-initialized { background: #1e3a5f; color: #64b5f6; }
    .badge-extracting  { background: #3a2a00; color: #ffd54f; }
    .badge-completed   { background: #1b3a2a; color: #66bb6a; }
    .badge-error       { background: #3a1a1a; color: #ef5350; }

    /* ---- Project list rows ---- */
    .project-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        background: #1a1a2a;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .project-name { font-weight: 600; font-size: 0.95rem; color: #ccd6f6; }
    .project-pdf  { font-size: 0.82rem; color: #8892b0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="cm-header">
    <h1>🎬 ComicMotion AI</h1>
    <p>Transform any comic PDF into a cinematic motion comic — locally, for free.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Layout: two columns
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.6], gap="large")

# ===========================================================================
# LEFT COLUMN — Create New Project
# ===========================================================================
with col_left:
    st.markdown("### ➕ Create New Project")

    with st.form("new_project_form", clear_on_submit=True):
        project_name_input = st.text_input(
            "Project Name",
            placeholder="e.g. Batman_Year_One",
            help="Letters, numbers, spaces and hyphens only. Spaces become underscores.",
        )
        pdf_upload = st.file_uploader(
            "Upload Comic File (PDF, CBZ, CBR)",
            type=None,           # We validate manually (extension + magic bytes)
            accept_multiple_files=False,
            help="Supported formats: PDF, CBZ (ZIP archive), CBR (RAR archive)",
        )
        submit = st.form_submit_button("🚀 Create Project", use_container_width=True)

    if submit:
        # --- Validate: project name ---
        if not project_name_input.strip():
            st.error("Please enter a project name.")
        elif pdf_upload is None:
            st.error("Please upload a PDF file.")
        else:
            try:
                safe_name = workspace.sanitize_project_name(project_name_input)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            # --- Validate: comic file (PDF / CBZ / CBR) ---
            is_valid, validation_msg, file_format = validator.validate_comic_file(pdf_upload)
            if not is_valid:
                st.error(validation_msg)
                st.stop()

            # --- Check duplicate ---
            if workspace.project_exists(safe_name):
                st.error(
                    f"A project named **{safe_name}** already exists. "
                    "Choose a different name."
                )
                st.stop()

            # --- Create workspace ---
            try:
                project_path = workspace.create_project_workspace(safe_name)
            except FileExistsError as e:
                st.error(str(e))
                st.stop()

            # --- Save uploaded file ---
            workspace.save_uploaded_pdf(safe_name, pdf_upload.read(), pdf_upload.name)

            # --- Initialize DB ---
            database.initialize_database(safe_name)
            database.insert_project(safe_name, pdf_upload.name, file_format)

            st.success(f"✅ Project **{safe_name}** created successfully!")
            st.rerun()

# ===========================================================================
# RIGHT COLUMN — Project Dashboard
# ===========================================================================
with col_right:
    st.markdown("### 📂 Projects")

    if "flash_success" in st.session_state:
        st.success(st.session_state["flash_success"])
        del st.session_state["flash_success"]

    existing_projects = workspace.list_projects()

    if not existing_projects:
        st.info("No projects yet. Create your first project on the left.", icon="💡")
    else:
        # Sort alphabetically
        for proj_name in sorted(existing_projects):
            project_record = database.get_project(proj_name)

            if project_record is None:
                # Workspace folder exists but DB was never written (partial failure)
                state = "UNKNOWN"
                pdf_name = "—"
                created = "—"
            else:
                state = project_record.get("state", "UNKNOWN")
                pdf_name = project_record.get("pdf_filename", "—")
                created = project_record.get("created_at", "—")

            # State → badge class mapping
            badge_class = {
                "INITIALIZED": "badge-initialized",
                "EXTRACTING":  "badge-extracting",
                "COMPLETED":   "badge-completed",
                "ERROR":       "badge-error",
            }.get(state, "badge-initialized")

            st.markdown(f"""
            <div class="project-row">
                <div>
                    <div class="project-name">{proj_name}</div>
                    <div class="project-pdf">📄 {pdf_name} &nbsp;|&nbsp; 🕐 {created}</div>
                </div>
                <span class="badge {badge_class}">{state}</span>
            </div>
            """, unsafe_allow_html=True)

    # ---- Selected project detail ----
    if existing_projects:
        st.markdown("---")
        st.markdown("### 🔍 Project Detail")
        selected = st.selectbox(
            "Select a project to inspect",
            options=sorted(existing_projects),
            label_visibility="collapsed",
        )
        if selected:
            record = database.get_project(selected)
            proj_path = workspace.get_project_path(selected)

            if record:
                # Project info table
                database.run_migrations(selected)
                record = database.get_project(selected)  # re-fetch after migration
                file_type_label = (record.get("file_type") or "pdf").upper()
                st.markdown(f"""
                <div class="cm-card">
                    <table style="width:100%; border-collapse:collapse;">
                        <tr>
                            <td style="color:#8892b0; padding:6px 0; width:140px;">Project Name</td>
                            <td style="color:#ccd6f6; font-weight:600;">{record['name']}</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0; padding:6px 0;">Source File</td>
                            <td style="color:#ccd6f6;">{record['pdf_filename']} <span style="color:#e94560;font-size:0.78rem;">[{file_type_label}]</span></td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0; padding:6px 0;">State</td>
                            <td><span class="badge badge-initialized">{record['state']}</span></td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0; padding:6px 0;">Created</td>
                            <td style="color:#ccd6f6;">{record['created_at']}</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0; padding:6px 0;">Extracted At</td>
                            <td style="color:#ccd6f6;">{record.get('extracted_at') or '—'}</td>
                        </tr>
                        <tr>
                            <td style="color:#8892b0; padding:6px 0;">Workspace</td>
                            <td style="color:#ccd6f6; font-size:0.82rem;">{proj_path}</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                # Folder structure check
                st.markdown("**Workspace Folder Structure**")
                subdirs = workspace.PROJECT_SUBDIRS
                cols = st.columns(3)
                for i, subdir in enumerate(subdirs):
                    folder_path = os.path.join(proj_path, subdir)
                    exists = os.path.isdir(folder_path)
                    icon = "✅" if exists else "❌"
                    cols[i % 3].markdown(f"{icon} `{subdir}/`")

                # ────────────────────────────────────────────────────────────
                # Extraction section
                # ────────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("**📸 Page Extraction**")

                # Ensure the DB schema is current for this project
                # (run_migrations already called above during project info fetch)

                project_state = record.get("state", "INITIALIZED")
                file_fmt = record.get("file_type") or "pdf"
                pages = database.get_all_pages(selected)
                extracted_pages = [p for p in pages if p["state"] == "EXTRACTED"]
                total_pages_db  = record.get("total_pages") or 0

                if project_state == "INITIALIZED":
                    pending_count = total_pages_db - len(extracted_pages) if total_pages_db else "?"

                    if pages and len(extracted_pages) > 0:
                        st.info(
                            f"Partial extraction detected: "
                            f"**{len(extracted_pages)}** done, "
                            f"**{pending_count}** remaining.",
                            icon="⚠️",
                        )
                        btn_label = "▶️ Resume Extraction"
                    else:
                        btn_label = "▶️ Extract Pages"

                    if st.button(btn_label, key=f"extract_{selected}", use_container_width=True):
                        prog_bar    = st.progress(0.0)
                        status_text = st.empty()

                        def _on_progress(current, total, page_num, status):
                            frac = current / total if total else 0
                            prog_bar.progress(frac)
                            icon_map = {
                                "extracted": "✅",
                                "skipped":   "⏭️",
                                "error":     "❌",
                            }
                            status_text.markdown(
                                f"{icon_map.get(status, '')} "
                                f"Page **{page_num}** of **{total}** — `{status}`"
                            )

                        with st.spinner("Extracting pages..."):
                            result = comic_ingestion.ingest(
                                selected,
                                file_fmt,
                                progress_callback=_on_progress,
                            )

                        if result["errors"]:
                            st.error(
                                f"Finished with **{len(result['errors'])}** error(s). "
                                "Check the details below."
                            )
                            for err in result["errors"]:
                                st.warning(f"Page {err['page']}: {err['error']}")
                        else:
                            st.success(
                                f"✅ Extraction complete! "
                                f"**{result['extracted']}** extracted, "
                                f"**{result['skipped']}** skipped (already done)."
                            )
                        st.rerun()

                elif project_state in ("EXTRACTED", "PANELS_READY"):
                    st.success(
                        f"✅ All **{len(extracted_pages)}** pages extracted.",
                        icon="✅",
                    )

                    # Show pages table
                    if extracted_pages:
                        st.markdown("**Extracted Pages**")
                        rows_html = "".join(
                            f"""
                            <tr>
                                <td style='padding:5px 10px;color:#ccd6f6;'>{p['page_number']}</td>
                                <td style='padding:5px 10px;color:#8892b0;font-size:0.82rem;'>{p['image_filename']}</td>
                                <td style='padding:5px 10px;color:#8892b0;'>{p['width']} × {p['height']}</td>
                                <td style='padding:5px 10px;color:#8892b0;font-size:0.78rem;'>{p['extracted_at']}</td>
                            </tr>
                            """
                            for p in extracted_pages[:20]  # cap at 20 rows in UI
                        )
                        st.markdown(
                            f"""
                            <div class='cm-card' style='max-height:340px;overflow-y:auto;'>
                            <table style='width:100%;border-collapse:collapse;'>
                                <thead>
                                    <tr style='border-bottom:1px solid rgba(255,255,255,0.08);'>
                                        <th style='text-align:left;padding:5px 10px;color:#e94560;font-size:0.82rem;'>#</th>
                                        <th style='text-align:left;padding:5px 10px;color:#e94560;font-size:0.82rem;'>File</th>
                                        <th style='text-align:left;padding:5px 10px;color:#e94560;font-size:0.82rem;'>Dimensions</th>
                                        <th style='text-align:left;padding:5px 10px;color:#e94560;font-size:0.82rem;'>Extracted At (UTC)</th>
                                    </tr>
                                </thead>
                                <tbody>{rows_html}</tbody>
                            </table>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if len(extracted_pages) > 20:
                            st.caption(
                                f"Showing first 20 of {len(extracted_pages)} pages."
                            )

                    # ────────────────────────────────────────────────────────────
                    # Panel Detection
                    # ────────────────────────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("**🔍 Panel Detection**")

                    if project_state == "EXTRACTED":
                        if st.button(
                            "🔍 Detect Panels",
                            key=f"detect_{selected}",
                            use_container_width=True,
                        ):
                            _pd_bar  = st.progress(0.0)
                            _pd_text = st.empty()

                            def _pd_callback(page_num, total, current, status):
                                _pd_bar.progress(current / total if total else 0)
                                _pd_icons = {
                                    "detected":  "✅",
                                    "no_panels": "⚠️",
                                    "error":     "❌",
                                }
                                _pd_text.markdown(
                                    f"{_pd_icons.get(status, '')} "
                                    f"Page **{page_num}** — `{status}`"
                                )

                            with st.spinner("Detecting panels…"):
                                _pd_result = panel_detector.detect_all_panels(
                                    selected,
                                    progress_callback=_pd_callback,
                                )

                            for _e in _pd_result["errors"]:
                                st.warning(_e)
                            if _pd_result["pages_with_issues"]:
                                st.warning(
                                    f"⚠️ {len(_pd_result['pages_with_issues'])} page(s) had "
                                    f"no detectable panels: {_pd_result['pages_with_issues']}"
                                )
                            st.success(
                                f"✅ Detection complete! "
                                f"**{_pd_result['total_panels']}** panels across "
                                f"**{_pd_result['pages_processed']}** pages."
                            )
                            st.rerun()

                    elif project_state == "PANELS_READY":
                        _pcount = database.count_panels(selected)
                        st.success(
                            f"✅ **{_pcount}** panels detected across "
                            f"**{len(extracted_pages)}** pages.",
                            icon="🔍",
                        )
                        if st.button(
                            "🔄 Re-detect Panels",
                            key=f"redetect_{selected}",
                            use_container_width=True,
                        ):
                            database.update_project_state(selected, "EXTRACTED")
                            st.rerun()

            else:
                st.warning("No database record found for this project.")

            # ────────────────────────────────────────────────────────────
            # Danger Zone: Delete Project
            # ────────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### ⚠️ Danger Zone")

            if f"delete_confirm_{selected}" not in st.session_state:
                st.session_state[f"delete_confirm_{selected}"] = False

            if st.session_state[f"delete_confirm_{selected}"]:
                st.warning(f"Delete project '{selected}'? This action cannot be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Cancel", key=f"cancel_del_{selected}", use_container_width=True):
                        st.session_state[f"delete_confirm_{selected}"] = False
                        st.rerun()
                with c2:
                    if st.button("Delete", key=f"confirm_del_{selected}", type="primary", use_container_width=True):
                        try:
                            workspace.delete_project_workspace(selected)
                            st.session_state["flash_success"] = "Project deleted successfully."
                            st.session_state[f"delete_confirm_{selected}"] = False
                            st.rerun()
                        except FileNotFoundError:
                            st.session_state["flash_success"] = "Project already deleted or folder missing."
                            st.session_state[f"delete_confirm_{selected}"] = False
                            st.rerun()
                        except PermissionError:
                            st.error("Permission denied. Close any files open in other programs and try again.")
                        except Exception as e:
                            st.error(f"Failed to delete project: {e}")
            else:
                if st.button("🗑️ Delete Project", key=f"btn_del_{selected}", use_container_width=True):
                    st.session_state[f"delete_confirm_{selected}"] = True
                    st.rerun()

# ===========================================================================
# FULL-WIDTH SECTION — Image Viewer
# ===========================================================================
st.markdown("---")
st.markdown("## 🖼️ Page Viewer")

_viewer_projects = workspace.list_projects()
if not _viewer_projects:
    st.info("Create and extract a project to use the viewer.", icon="💡")
else:
    _vsel = st.selectbox(
        "Select project to view",
        options=sorted(_viewer_projects),
        key="viewer_project_select",
    )
    if _vsel:
        _vpages = database.get_all_pages(_vsel)
        _extracted = [p for p in _vpages if p["state"] == "EXTRACTED"]

        if not _extracted:
            st.info(
                f"Project **{_vsel}** has no extracted pages yet. "
                "Run extraction first.",
                icon="📷",
            )
        else:
            _total = len(_extracted)

            # Session-state key per project keeps navigation state isolated
            _sk = f"viewer_idx_{_vsel}"
            if _sk not in st.session_state:
                st.session_state[_sk] = 0

            # Clamp index in case project changed
            st.session_state[_sk] = max(0, min(st.session_state[_sk], _total - 1))
            _idx = st.session_state[_sk]
            _cur_page = _extracted[_idx]

            # ── Navigation bar ───────────────────────────────────────────────
            nav_cols = st.columns([1, 1, 4, 1, 1])
            with nav_cols[0]:
                if st.button("⏮ First", use_container_width=True, key="vw_first"):
                    st.session_state[_sk] = 0
                    st.rerun()
            with nav_cols[1]:
                if st.button("◀ Prev", use_container_width=True, key="vw_prev"):
                    if _idx > 0:
                        st.session_state[_sk] = _idx - 1
                    st.rerun()
            with nav_cols[2]:
                _jump = st.slider(
                    "Page",
                    min_value=1,
                    max_value=_total,
                    value=_idx + 1,
                    key="vw_slider",
                    label_visibility="collapsed",
                )
                if _jump - 1 != _idx:
                    st.session_state[_sk] = _jump - 1
                    st.rerun()
            with nav_cols[3]:
                if st.button("Next ▶", use_container_width=True, key="vw_next"):
                    if _idx < _total - 1:
                        st.session_state[_sk] = _idx + 1
                    st.rerun()
            with nav_cols[4]:
                if st.button("Last ⏭", use_container_width=True, key="vw_last"):
                    st.session_state[_sk] = _total - 1
                    st.rerun()

            # ── Page info + preview ──────────────────────────────────────────
            info_col, preview_col = st.columns([1, 3], gap="large")

            with info_col:
                st.markdown(f"""
                <div class="cm-card">
                    <table style="width:100%;border-collapse:collapse;">
                        <tr><td style="color:#8892b0;padding:5px 0;">Page</td>
                            <td style="color:#ccd6f6;font-weight:700;font-size:1.4rem;">
                                {_cur_page['page_number']} / {_total}</td></tr>
                        <tr><td style="color:#8892b0;padding:5px 0;">File</td>
                            <td style="color:#ccd6f6;font-size:0.82rem;">{_cur_page['image_filename']}</td></tr>
                        <tr><td style="color:#8892b0;padding:5px 0;">Width</td>
                            <td style="color:#ccd6f6;">{_cur_page['width']} px</td></tr>
                        <tr><td style="color:#8892b0;padding:5px 0;">Height</td>
                            <td style="color:#ccd6f6;">{_cur_page['height']} px</td></tr>
                        <tr><td style="color:#8892b0;padding:5px 0;">Extracted</td>
                            <td style="color:#8892b0;font-size:0.78rem;">{_cur_page['extracted_at']}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                # ── Thumbnail strip (pages around current) ───────────────────
                st.markdown("**Nearby Pages**")
                _strip_start = max(0, _idx - 3)
                _strip_end   = min(_total, _idx + 4)
                _strip_pages = _extracted[_strip_start:_strip_end]

                for _sp in _strip_pages:
                    _sp_idx = _sp["page_number"] - 1
                    _is_current = (_sp_idx == _idx)
                    _border = "2px solid #e94560" if _is_current else "2px solid transparent"
                    _img_path = _sp.get("image_path")
                    if _img_path and os.path.exists(_img_path):
                        st.image(
                            _img_path,
                            caption=f"p.{_sp['page_number']}",
                            width=120,
                        )
                        if not _is_current:
                            if st.button(
                                f"Go to {_sp['page_number']}",
                                key=f"thumb_{_vsel}_{_sp_idx}",
                                use_container_width=True,
                            ):
                                st.session_state[_sk] = _sp_idx
                                st.rerun()

            with preview_col:
                _img_path = _cur_page.get("image_path")
                if _img_path and os.path.exists(_img_path):
                    st.image(
                        _img_path,
                        caption=f"Page {_cur_page['page_number']} — "
                                f"{_cur_page['width']}x{_cur_page['height']}px",
                        use_column_width=True,
                    )
                else:
                    st.warning(
                        f"Image file not found: {_img_path}. "
                        "Re-run extraction to regenerate it."
                    )

# ===========================================================================
# FULL-WIDTH SECTION — Panel Viewer
# ===========================================================================
st.markdown("---")
st.markdown("## 🗹 Panel Viewer")

_pv_projs = workspace.list_projects()
if not _pv_projs:
    st.info("No projects found. Create a project and run extraction first.", icon="💡")
else:
    _pv_proj = st.selectbox(
        "Project",
        options=sorted(_pv_projs),
        key="pv_project_select",
    )
    if _pv_proj:
        _pv_all_pages = database.get_all_pages(_pv_proj)
        _pv_pages = [p for p in _pv_all_pages if p["state"] == "EXTRACTED"]

        if not _pv_pages:
            st.info(
                f"Project **{_pv_proj}** has no extracted pages yet.",
                icon="📷",
            )
        else:
            # Build mapping: page_number -> (page_record, [panels])
            _pv_page_map: dict[int, tuple[dict, list]] = {}
            for _pg in _pv_pages:
                _pv_page_map[_pg["page_number"]] = (
                    _pg,
                    database.get_panels_for_page(_pv_proj, _pg["id"]),
                )

            _pv_page_num = st.selectbox(
                "Page",
                options=sorted(_pv_page_map.keys()),
                format_func=lambda n: (
                    f"Page {n} — {len(_pv_page_map[n][1])} panel(s)"
                ),
                key="pv_page_select",
            )

            if _pv_page_num is not None:
                _pv_page_rec, _pv_panels = _pv_page_map[_pv_page_num]

                if not _pv_panels:
                    st.warning(
                        f"No panels detected on page {_pv_page_num}. "
                        "Run **Panel Detection** in the project detail above.",
                        icon="⚠️",
                    )
                else:
                    _pv_ann_col, _pv_thumb_col = st.columns([2, 1], gap="large")

                    # ── Left: annotated page ──────────────────────────────
                    with _pv_ann_col:
                        st.markdown(
                            f"**Page {_pv_page_num} — "
                            f"{len(_pv_panels)} panel(s) detected**"
                        )
                        _pv_img_path = _pv_page_rec.get("image_path")
                        if _pv_img_path and os.path.exists(_pv_img_path):
                            try:
                                _pv_overlay = panel_detector.draw_panel_overlays(
                                    _pv_img_path, _pv_panels
                                )
                                st.image(
                                    _pv_overlay,
                                    caption=(
                                        f"Page {_pv_page_num} — panel boundaries"
                                    ),
                                    use_column_width=True,
                                )
                            except Exception as _pv_err:
                                st.error(f"Cannot render overlay: {_pv_err}")
                                st.image(_pv_img_path, use_column_width=True)
                        else:
                            st.warning("Original page image not found on disk.")

                    # ── Right: thumbnails + selected-panel preview ────────
                    with _pv_thumb_col:
                        st.markdown("**Panels — click to preview**")

                        _pv_sel_key = f"pv_sel_{_pv_proj}_{_pv_page_num}"
                        if _pv_sel_key not in st.session_state:
                            st.session_state[_pv_sel_key] = None

                        for _pv_p in _pv_panels:
                            _pi = _pv_p["panel_index"]
                            _pp = _pv_p.get("image_path", "")
                            if _pp and os.path.exists(_pp):
                                _pbbox = json.loads(_pv_p["bounding_box"])
                                st.image(_pp, width=140,
                                         caption=f"Panel {_pi}")
                                st.caption(
                                    f"{_pv_p['width']}×{_pv_p['height']} px | "
                                    f"box ({_pbbox['x']},{_pbbox['y']}) "
                                    f"{_pbbox['w']}×{_pbbox['h']}"
                                )
                                if st.button(
                                    f"Preview panel {_pi}",
                                    key=f"pvbtn_{_pv_proj}_{_pv_page_num}_{_pi}",
                                    use_container_width=True,
                                ):
                                    st.session_state[_pv_sel_key] = _pi
                            else:
                                st.warning(f"Panel {_pi} file missing.")

                        # ── Large preview of selected panel ────────────
                        _pv_chosen = st.session_state.get(_pv_sel_key)
                        if _pv_chosen is not None:
                            _pv_sel_p = next(
                                (
                                    p for p in _pv_panels
                                    if p["panel_index"] == _pv_chosen
                                ),
                                None,
                            )
                            if _pv_sel_p:
                                _pv_sp_path = _pv_sel_p.get("image_path", "")
                                if _pv_sp_path and os.path.exists(_pv_sp_path):
                                    st.markdown("---")
                                    st.markdown(
                                        f"**Panel {_pv_chosen} — full preview**"
                                    )
                                    _sbb = json.loads(_pv_sel_p["bounding_box"])
                                    st.caption(
                                        f"Size: {_pv_sel_p['width']}×"
                                        f"{_pv_sel_p['height']} px | "
                                        f"Position on page: x={_sbb['x']}, "
                                        f"y={_sbb['y']}, "
                                        f"w={_sbb['w']}, h={_sbb['h']}"
                                    )
                                    st.image(
                                        _pv_sp_path,
                                        use_column_width=True,
                                    )

# ===========================================================================
# FULL-WIDTH SECTION — Production Pipeline (Modules 4–8)
# ===========================================================================
st.markdown("---")
st.markdown("## Production Pipeline")
_pipe_projects = workspace.list_projects()
if not _pipe_projects:
    st.info("Create a project and detect panels to begin the production pipeline.")
else:
    _pipe_project = st.selectbox("Project for analysis and rendering", sorted(_pipe_projects), key="pipeline_project")
    database.run_migrations(_pipe_project)
    _pipe_record = database.get_project(_pipe_project)
    _pipe_panels = database.get_all_panels(_pipe_project)
    _pipe_story = database.get_story_sequences(_pipe_project)
    _pipe_col1, _pipe_col2, _pipe_col3, _pipe_col4, _pipe_col5 = st.columns(5)
    with _pipe_col1:
        if st.button("1. Analyze Panels", use_container_width=True):
            try:
                if not os.getenv("GEMINI_API_KEY"):
                    st.error("Set the GEMINI_API_KEY environment variable, restart Streamlit, then try again.")
                    st.stop()
                _bar = st.progress(0.0)
                _result = gemini_vision.analyze_all_panels(_pipe_project, lambda current, total, _: _bar.progress(current / total))
                if _result["errors"]:
                    st.warning(f"Analyzed {_result['analyzed']} panels; {len(_result['errors'])} failed. Re-run analysis to resume failed panels.")
                    with st.expander("Analysis error details"):
                        for _error in _result["errors"][:20]: st.code(_error)
                        if len(_result["errors"]) > 20: st.caption(f"Showing 20 of {len(_result['errors'])} errors.")
                else:
                    _story = story_manager.build_story(_pipe_project)
                    st.success(f"Analyzed {_result['analyzed']} panels with {_result['model']}; created {_story['sequences']} story beats.")
            except Exception as exc: st.error(str(exc))
    with _pipe_col2:
        if st.button("2. Build Story", use_container_width=True):
            try:
                _result = story_manager.build_story(_pipe_project)
                st.success(f"Built {_result['sequences']} story beats.")
            except Exception as exc: st.error(str(exc))
    with _pipe_col3:
        if st.button("3. Generate Audio", use_container_width=True):
            try:
                _bar = st.progress(0.0)
                _result = tts_service.generate_audio(_pipe_project, progress_callback=lambda current, total: _bar.progress(current / total))
                if _result["errors"]: st.error("\n".join(_result["errors"]))
                else: st.success(f"Generated {_result['generated']} audio tracks.")
            except Exception as exc: st.error(str(exc))
    with _pipe_col4:
        if st.button("4. Render Clips", use_container_width=True):
            try:
                _bar = st.progress(0.0)
                _result = motion_renderer.render_motion_clips(_pipe_project, progress_callback=lambda current, total: _bar.progress(current / total))
                if _result["errors"]: st.error("\n".join(_result["errors"]))
                else: st.success(f"Rendered {_result['rendered']} clips.")
            except Exception as exc: st.error(str(exc))
    with _pipe_col5:
        if st.button("5. Assemble MP4", use_container_width=True):
            try:
                _bar = st.progress(0.0)
                _output = final_renderer.assemble_video(_pipe_project, progress_callback=lambda current, total: _bar.progress(current / total))
                st.success("Final video exported.")
                st.video(_output)
            except Exception as exc: st.error(str(exc))
    st.caption(f"State: {_pipe_record.get('state', 'UNKNOWN')} | Panels: {len(_pipe_panels)} | Story beats: {len(_pipe_story)}")

st.markdown("---")
st.markdown("## Panel Detection Debug")
_debug_projects = workspace.list_projects()
if _debug_projects:
    if st.checkbox("Enable debug mode (runs only for one selected page)", key="panel_debug_enabled"):
        _debug_project = st.selectbox("Debug project", sorted(_debug_projects), key="panel_debug_project")
        _debug_pages = [page for page in database.get_all_pages(_debug_project) if page.get("image_path") and os.path.exists(page["image_path"])]
        if _debug_pages:
            _debug_page = st.selectbox("Debug page", _debug_pages, format_func=lambda page: f"Page {page['page_number']}", key="panel_debug_page")
            if st.button("Run panel detection debug", key="run_panel_debug"):
                try:
                    _debug = panel_detector.detect_panels_debug(_debug_page["image_path"])
                    _m1, _m2, _m3 = st.columns(3)
                    _m1.metric("Contours found", _debug["contours_found"])
                    _m2.metric("Accepted panels", len(_debug["accepted"]))
                    _m3.metric("Rejected contours", len(_debug["rejected"]))
                    _d1, _d2, _d3, _d4 = st.columns(4)
                    _d1.image(_debug["original"], caption="Original page", channels="BGR")
                    _d2.image(_debug["grayscale"], caption="Grayscale")
                    _d3.image(_debug["binary"], caption="Binary")
                    _d4.image(_debug["morphological"], caption="Morphological result")
                    _d5, _d6, _d7 = st.columns(3)
                    _d5.image(_debug["all_contours"], caption="All detected contours", channels="BGR")
                    _d6.image(_debug["accepted_contours"], caption="Accepted contours", channels="BGR")
                    _d7.image(_debug["rejected_contours"], caption="Rejected contours", channels="BGR")
                    if _debug["rejected"]:
                        st.dataframe(_debug["rejected"], use_container_width=True)
                except Exception as _debug_error:
                    st.error(f"Debug failed: {_debug_error}")
