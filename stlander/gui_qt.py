from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from vtk import vtkLight, vtkCamera

from PySide6 import QtCore, QtWidgets

from .core import (
    load_mesh,
    align_to_principal_axes,
    save_mesh,
    AlignmentResult,
    rotate_alignment_result,
)

@dataclass
class ViewStyle:
    projection: str = "Parallel"   # or "Perspective"
    show_edges: bool = False
    show_axes: bool = True
    show_outline: bool = False
    link_views: bool = True
    ambient: float = 0.3
    diffuse: float = 0.6
    specular: float = 0.4


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STLander — Principal Axes Aligner")
        self.resize(1400, 800)

        self._mesh_path: Optional[str] = None
        self._original: Optional[pv.PolyData] = None
        self._result: Optional[AlignmentResult] = None

        self.style = ViewStyle()
        self._syncing = False

        self._build_ui()
        self._install_camera_sync()
        self._apply_lighting()
        # Start the application maximized (fill the screen)
        self.showMaximized()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Splitter for render views
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        self.view_left = QtInteractor(self)
        self.view_right = QtInteractor(self)

        splitter.addWidget(self.view_left.interactor)
        splitter.addWidget(self.view_right.interactor)
        splitter.setSizes([700, 700])

        # Controls panel
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setMinimumWidth(320)
        root.addWidget(panel, 0)

        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        # File controls
        row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load STL…")
        self.btn_save = QtWidgets.QPushButton("Save aligned…")
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_save)
        v.addLayout(row)

        self.btn_reset = QtWidgets.QPushButton("Reset view")
        v.addWidget(self.btn_reset)

        v.addWidget(_hline())

        # Principal axis mapping
        v.addWidget(QtWidgets.QLabel("Principal axis mapping"))
        self.cmb_pa2 = QtWidgets.QComboBox()
        self.cmb_pa2.addItems(["PA2 -> Y (default)", "PA2 -> Z (swap Y/Z)"])
        self.cmb_pa2.setCurrentIndex(0)
        v.addWidget(self.cmb_pa2)

        # Rotation buttons (180°)
        v.addWidget(QtWidgets.QLabel("Rotate 180° around axis"))
        rot_row = QtWidgets.QHBoxLayout()
        self.btn_rot_x = QtWidgets.QPushButton("Rotate X")
        self.btn_rot_y = QtWidgets.QPushButton("Rotate Y")
        self.btn_rot_z = QtWidgets.QPushButton("Rotate Z")
        rot_row.addWidget(self.btn_rot_x)
        rot_row.addWidget(self.btn_rot_y)
        rot_row.addWidget(self.btn_rot_z)
        v.addLayout(rot_row)

        v.addWidget(_hline())

        # Tab widget for controls
        tabs = QtWidgets.QTabWidget()
        v.addWidget(tabs, 1)

        # ===== Tab 1: View Controls =====
        view_controls_tab = QtWidgets.QWidget()
        view_ctrl_layout = QtWidgets.QVBoxLayout(view_controls_tab)
        view_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        view_ctrl_layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        
        self.cmb_projection = QtWidgets.QComboBox()
        self.cmb_projection.addItems(["Parallel", "Perspective"])
        self.cmb_projection.setCurrentText(self.style.projection)
        form.addRow("Projection", self.cmb_projection)

        view_ctrl_layout.addLayout(form)
        view_ctrl_layout.addWidget(_hline())

        self.chk_edges = QtWidgets.QCheckBox("Show edges")
        self.chk_edges.setChecked(self.style.show_edges)
        self.chk_axes = QtWidgets.QCheckBox("Show axes widget")
        self.chk_axes.setChecked(self.style.show_axes)
        self.chk_outline = QtWidgets.QCheckBox("Show outline")
        self.chk_outline.setChecked(self.style.show_outline)
        self.chk_link = QtWidgets.QCheckBox("Link views")
        self.chk_link.setChecked(self.style.link_views)

        view_ctrl_layout.addWidget(self.chk_edges)
        view_ctrl_layout.addWidget(self.chk_axes)
        view_ctrl_layout.addWidget(self.chk_outline)
        view_ctrl_layout.addWidget(self.chk_link)

        view_ctrl_layout.addWidget(_hline())

        # (Opacity control removed)

        view_ctrl_layout.addWidget(_hline())

        # View along axis
        view_ctrl_layout.addWidget(QtWidgets.QLabel("View along axis"))
        grid = QtWidgets.QGridLayout()
        self.btn_px = QtWidgets.QPushButton("+X")
        self.btn_nx = QtWidgets.QPushButton("−X")
        self.btn_py = QtWidgets.QPushButton("+Y")
        self.btn_ny = QtWidgets.QPushButton("−Y")
        self.btn_pz = QtWidgets.QPushButton("+Z")
        self.btn_nz = QtWidgets.QPushButton("−Z")
        grid.addWidget(self.btn_px, 0, 0)
        grid.addWidget(self.btn_nx, 0, 1)
        grid.addWidget(self.btn_py, 1, 0)
        grid.addWidget(self.btn_ny, 1, 1)
        grid.addWidget(self.btn_pz, 2, 0)
        grid.addWidget(self.btn_nz, 2, 1)
        view_ctrl_layout.addLayout(grid)

        # Isometric view
        self.btn_isometric = QtWidgets.QPushButton("Isometric")
        view_ctrl_layout.addWidget(self.btn_isometric)

        view_ctrl_layout.addStretch(1)
        tabs.addTab(view_controls_tab, "View Controls")

        # ===== Tab 2: Mesh Info =====
        info_tab = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_tab)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(8)

        # Info box
        self.lbl_path = QtWidgets.QLabel("No file loaded.")
        self.lbl_path.setWordWrap(True)

        self.lbl_com = QtWidgets.QLabel("COM: —")
        self.lbl_evals = QtWidgets.QLabel("Eigenvalues: —")
        self.txt_axes = QtWidgets.QPlainTextEdit()
        self.txt_axes.setReadOnly(True)
        self.txt_axes.setMaximumHeight(120)
        self.txt_axes.setPlaceholderText("Axes (columns) will appear here…")

        info_layout.addWidget(self.lbl_path)
        info_layout.addWidget(self.lbl_com)
        info_layout.addWidget(self.lbl_evals)
        info_layout.addWidget(QtWidgets.QLabel("Axes (columns):"))
        info_layout.addWidget(self.txt_axes)
        info_layout.addStretch(1)
        tabs.addTab(info_tab, "Mesh Info")

        # ===== Tab 3: Lighting =====
        lighting_tab = QtWidgets.QWidget()
        lighting_layout = QtWidgets.QVBoxLayout(lighting_tab)
        lighting_layout.setContentsMargins(0, 0, 0, 0)
        lighting_layout.setSpacing(8)

        # Three-light preset button
        self.btn_three_lights = QtWidgets.QPushButton("Apply Three-Light Setup")
        lighting_layout.addWidget(self.btn_three_lights)
        lighting_layout.addWidget(_hline())

        # Ambient lighting
        self.sld_ambient = _slider_int(0, 100, int(self.style.ambient * 100))
        self.lbl_ambient = QtWidgets.QLabel(f"{self.style.ambient:.2f}")
        ambient_row = QtWidgets.QHBoxLayout()
        ambient_row.addWidget(self.sld_ambient, 1)
        ambient_row.addWidget(self.lbl_ambient, 0)
        lighting_layout.addWidget(_labeled_row("Ambient", ambient_row))

        # Diffuse lighting
        self.sld_diffuse = _slider_int(0, 100, int(self.style.diffuse * 100))
        self.lbl_diffuse = QtWidgets.QLabel(f"{self.style.diffuse:.2f}")
        diffuse_row = QtWidgets.QHBoxLayout()
        diffuse_row.addWidget(self.sld_diffuse, 1)
        diffuse_row.addWidget(self.lbl_diffuse, 0)
        lighting_layout.addWidget(_labeled_row("Diffuse", diffuse_row))

        # Specular lighting
        self.sld_specular = _slider_int(0, 100, int(self.style.specular * 100))
        self.lbl_specular = QtWidgets.QLabel(f"{self.style.specular:.2f}")
        specular_row = QtWidgets.QHBoxLayout()
        specular_row.addWidget(self.sld_specular, 1)
        specular_row.addWidget(self.lbl_specular, 0)
        lighting_layout.addWidget(_labeled_row("Specular", specular_row))

        lighting_layout.addStretch(1)
        tabs.addTab(lighting_tab, "Lighting")

        # Status bar with progress
        self.status = QtWidgets.QLabel("Ready.")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addWidget(self.progress_bar, 0)

        # Signals
        self.btn_load.clicked.connect(self.on_load)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_reset.clicked.connect(self.on_reset_view)

        self.cmb_projection.currentTextChanged.connect(self.on_style_changed)
        self.chk_edges.toggled.connect(self.on_style_changed)
        self.chk_axes.toggled.connect(self.on_style_changed)
        self.chk_outline.toggled.connect(self.on_style_changed)
        self.chk_link.toggled.connect(self.on_style_changed)
        self.sld_ambient.valueChanged.connect(self.on_ambient_changed)
        self.sld_diffuse.valueChanged.connect(self.on_diffuse_changed)
        self.sld_specular.valueChanged.connect(self.on_specular_changed)
        self.btn_three_lights.clicked.connect(self.on_three_lights)
        self.cmb_pa2.currentIndexChanged.connect(self.on_pa2_changed)
        self.btn_px.clicked.connect(lambda: self.on_view_axis("+X"))
        self.btn_nx.clicked.connect(lambda: self.on_view_axis("-X"))
        self.btn_py.clicked.connect(lambda: self.on_view_axis("+Y"))
        self.btn_ny.clicked.connect(lambda: self.on_view_axis("-Y"))
        self.btn_pz.clicked.connect(lambda: self.on_view_axis("+Z"))
        self.btn_nz.clicked.connect(lambda: self.on_view_axis("-Z"))
        self.btn_isometric.clicked.connect(self.on_isometric_view)
       
       # Rotation buttons
        self.btn_rot_x.clicked.connect(lambda: self.on_rotate_axis("X"))
        self.btn_rot_y.clicked.connect(lambda: self.on_rotate_axis("Y"))
        self.btn_rot_z.clicked.connect(lambda: self.on_rotate_axis("Z"))

        # Initial empty views
        self._init_views()

    def _init_views(self) -> None:
        for view, title in [(self.view_left, "Original"
        ""), (self.view_right, "Aligned")]:
            view.set_background("white")
            view.add_text(title, font_size=12)
            view.show_bounds(all_edges=True)
            view.reset_camera()

    # ---------- Camera sync ----------
    def _install_camera_sync(self) -> None:
        # Observe camera modifications on both sides and mirror to the other
        self.view_left.camera.AddObserver("ModifiedEvent", self._on_left_camera)
        self.view_right.camera.AddObserver("ModifiedEvent", self._on_right_camera)

    def _copy_camera(self, src: QtInteractor, dst: QtInteractor) -> None:
        # Avoid loops
        if self._syncing or not self.style.link_views:
            return
        self._syncing = True
        try:
            # camera_position includes (position, focal_point, up)
            dst.camera_position = src.camera_position
            # also keep projection settings consistent
            dst.camera.parallel_projection = src.camera.parallel_projection
            dst.camera.view_angle = src.camera.view_angle
            dst.camera.parallel_scale = src.camera.parallel_scale
            dst.render()
        finally:
            self._syncing = False

    def _on_left_camera(self, *args) -> None:
        self._copy_camera(self.view_left, self.view_right)

    def _on_right_camera(self, *args) -> None:
        self._copy_camera(self.view_right, self.view_left)

    # ---------- Plot updating ----------
    def _apply_projection(self, view: QtInteractor) -> None:
        if self.style.projection == "Parallel":
            view.enable_parallel_projection()
        else:
            view.disable_parallel_projection()

    def _add_mesh_with_style(self, view: QtInteractor, mesh: pv.PolyData) -> None:
        actor = view.add_mesh(
            mesh,
            opacity=1.0,
            show_edges=bool(self.style.show_edges),
            style="surface",
        )

        # Apply default Gouraud shading
        if actor is not None:
            try:
                prop = actor.GetProperty()
                prop.SetInterpolationToGouraud()
            except Exception:
                pass

    def _refresh_views(self) -> None:
        # Clear + redraw based on current state + style
        self.view_left.clear()
        self.view_right.clear()

        self._apply_projection(self.view_left)
        self._apply_projection(self.view_right)

        self.view_left.add_text("Original", font_size=12)
        self.view_right.add_text("Aligned", font_size=12)

        if self._original is not None:
            self._add_mesh_with_style(self.view_left, self._original)

        if self._result is not None:
            self._add_mesh_with_style(self.view_right, self._result.aligned)
        else:
            self.view_right.add_text("Load + Transform to view", position="lower_left", font_size=10)

        if self.style.show_axes:
            self.view_left.add_axes()
            self.view_right.add_axes()
        else:
            self.view_left.hide_axes()
            self.view_right.hide_axes()

        if self.style.show_outline:
            self.view_left.show_bounds(all_edges=True)
            self.view_right.show_bounds(all_edges=True)
        else:
            self.view_left.remove_bounds_axes()
            self.view_right.remove_bounds_axes()

        # If linking, match cameras once after redraw
        if self.style.link_views and self._original is not None:
            self._copy_camera(self.view_left, self.view_right)


        self.view_left.render()
        self.view_right.render()
        self._apply_lighting()

    def _set_progress(self, value: int, message: str = "") -> None:
        """Update progress bar and status message."""
        self.progress_bar.setValue(value)
        if message:
            self.status.setText(message)
        if value >= 100:
            self.progress_bar.setVisible(False)

    def _show_progress(self, message: str = "Processing...") -> None:
        """Show progress bar and set initial message."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status.setText(message)

    # ---------- Slots ----------
    @QtCore.Slot()
    def on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open STL",
            "",
            "STL files (*.stl);;All files (*.*)",
        )
        if not path:
            return
        
        self._show_progress("Loading mesh...")
        self._set_progress(10, "Loading mesh...")
        
        try:
            mesh = load_mesh(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            self.progress_bar.setVisible(False)
            self.status.setText("Load failed.")
            return

        self._set_progress(50, "Processing mesh...")
        self._mesh_path = path
        self._original = mesh

        self.lbl_path.setText(f"Loaded: {path}")
        
        # Automatically align to principal axes
        self._set_progress(60, "Computing principal axes...")
        try:
            res = align_to_principal_axes(self._original, pa2_target=self._pa2_target())
            self._result = res
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Alignment failed", str(e))
            self._result = None
            self.progress_bar.setVisible(False)
            self.status.setText("Alignment failed.")
            return

        com = res.center_of_mass
        evals = res.eigenvalues
        axes = res.axes_matrix
        self.lbl_com.setText(f"COM: [{com[0]: .6g}, {com[1]: .6g}, {com[2]: .6g}]")
        self.lbl_evals.setText(f"Eigenvalues: [{evals[0]: .6g}, {evals[1]: .6g}, {evals[2]: .6g}]")
        self.txt_axes.setPlainText(np.array2string(axes, precision=4, suppress_small=True))
        
        self._set_progress(85, "Refreshing views...")
        self._refresh_views()
        
        self._set_progress(100, "Loaded and aligned. Views are linked (toggle in controls).")
        self.view_left.reset_camera()
        self.view_right.reset_camera()
        self.view_left.render()
        self.view_right.render()

    @QtCore.Slot()
    def on_save(self) -> None:
        if self._result is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Run Transform first.")
            return

        default_name = "aligned.stl"
        initial_dir = ""
        if self._mesh_path:
            root, _ = os.path.splitext(os.path.basename(self._mesh_path))
            default_name = f"{root}_aligned.stl"
            initial_dir = os.path.dirname(self._mesh_path)

        # Provide initial directory + filename so dialog opens in original folder
        start_path = os.path.join(initial_dir, default_name) if initial_dir else default_name
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save aligned mesh",
            start_path,
            "STL files (*.stl);;All files (*.*)",
        )
        if not path:
            return
        
        self._show_progress("Saving mesh...")
        self._set_progress(30, "Writing file...")
        
        try:
            save_mesh(self._result.aligned, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
            self.progress_bar.setVisible(False)
            self.status.setText("Save failed.")
            return

        self._set_progress(100, f"Saved: {path}")

    @QtCore.Slot()
    def on_reset_view(self) -> None:
        self.view_left.reset_camera()
        self.view_right.reset_camera()
        self.view_left.render()
        self.view_right.render()

    @QtCore.Slot()
    def on_style_changed(self) -> None:
        self.style.projection = self.cmb_projection.currentText()
        self.style.show_edges = self.chk_edges.isChecked()
        self.style.show_axes = self.chk_axes.isChecked()
        self.style.show_outline = self.chk_outline.isChecked()
        self.style.link_views = self.chk_link.isChecked()
        self._refresh_views()

    @QtCore.Slot(int)
    def on_opacity_changed(self, v: int) -> None:
        self.style.opacity = float(v) / 100.0
        self.lbl_opacity.setText(f"{self.style.opacity:.2f}")
        self._refresh_views()

    @QtCore.Slot(int)
    def on_ambient_changed(self, v: int) -> None:
        self.style.ambient = float(v) / 100.0
        self.lbl_ambient.setText(f"{self.style.ambient:.2f}")
        self._apply_lighting()

    @QtCore.Slot(int)
    def on_diffuse_changed(self, v: int) -> None:
        self.style.diffuse = float(v) / 100.0
        self.lbl_diffuse.setText(f"{self.style.diffuse:.2f}")
        self._apply_lighting()

    @QtCore.Slot(int)
    def on_specular_changed(self, v: int) -> None:
        self.style.specular = float(v) / 100.0
        self.lbl_specular.setText(f"{self.style.specular:.2f}")
        self._apply_lighting()


    def _apply_lighting(self) -> None:
        """Apply lighting settings to both views using VTK light objects."""
        for view in [self.view_left, self.view_right]:
            try:
                renderer = view.renderer
                if renderer is None:
                    continue
                
                # Remove existing lights to avoid accumulation
                light_collection = renderer.GetLights()
                light_collection.RemoveAllItems()
                
                # Create three lights with configurable intensities
                # Key light (main light from front-right)
                key_light = vtkLight()
                key_light.SetPosition(1.0, 1.0, 1.0)
                key_light.SetFocalPoint(0.0, 0.0, 0.0)
                key_light.SetIntensity(self.style.diffuse)
                key_light.SetColor(1.0, 1.0, 1.0)
                renderer.AddLight(key_light)
                
                # Fill light (softer light from the left)
                fill_light = vtkLight()
                fill_light.SetPosition(-1.0, 0.5, 0.5)
                fill_light.SetFocalPoint(0.0, 0.0, 0.0)
                fill_light.SetIntensity(self.style.ambient)
                fill_light.SetColor(0.9, 0.9, 1.0)
                renderer.AddLight(fill_light)
                
                # Back light (rim/separation light)
                back_light = vtkLight()
                back_light.SetPosition(0.0, -1.0, 1.0)
                back_light.SetFocalPoint(0.0, 0.0, 0.0)
                back_light.SetIntensity(self.style.specular)
                back_light.SetColor(1.0, 1.0, 0.9)
                renderer.AddLight(back_light)
                
                # Set ambient color for base illumination
                renderer.SetAmbient(0.2, 0.2, 0.2)
                
            except Exception:
                pass  # Silently handle if lighting application fails
        
        self.view_left.render()
        self.view_right.render()

    @QtCore.Slot()
    def on_three_lights(self) -> None:
        """Apply professional three-light setup: key light, fill light, and back light."""
        # Three-light configuration: bright key and fill, good specular for highlights
        self.style.ambient = 0.4   # Fill light (overhead)
        self.style.diffuse = 0.8   # Key light (main illumination)
        self.style.specular = 0.6  # Back light (highlights and rim)
        
        # Update sliders and labels
        self.sld_ambient.blockSignals(True)
        self.sld_diffuse.blockSignals(True)
        self.sld_specular.blockSignals(True)
        
        self.sld_ambient.setValue(int(self.style.ambient * 100))
        self.sld_diffuse.setValue(int(self.style.diffuse * 100))
        self.sld_specular.setValue(int(self.style.specular * 100))
        
        self.lbl_ambient.setText(f"{self.style.ambient:.2f}")
        self.lbl_diffuse.setText(f"{self.style.diffuse:.2f}")
        self.lbl_specular.setText(f"{self.style.specular:.2f}")
        
        self.sld_ambient.blockSignals(False)
        self.sld_diffuse.blockSignals(False)
        self.sld_specular.blockSignals(False)
        
        self._apply_lighting()

    def _pa2_target(self) -> str:
        # Map UI selection to core parameter
        if getattr(self, "cmb_pa2", None) is not None and self.cmb_pa2.currentIndex() == 1:
            return "Z"
        return "Y"

    @QtCore.Slot()
    def on_pa2_changed(self) -> None:
        # If we already have a result, re-run alignment with the new mapping
        if self._original is None or self._result is None:
            return
        try:
            res = align_to_principal_axes(self._original, pa2_target=self._pa2_target())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Re-align failed", str(e))
            return

        self._result = res
        com = res.center_of_mass
        evals = res.eigenvalues
        axes = res.axes_matrix
        self.lbl_com.setText(f"COM: [{com[0]: .6g}, {com[1]: .6g}, {com[2]: .6g}]")
        self.lbl_evals.setText(f"Eigenvalues: [{evals[0]: .6g}, {evals[1]: .6g}, {evals[2]: .6g}]")
        self.txt_axes.setPlainText(np.array2string(axes, precision=4, suppress_small=True))
        self.status.setText("Re-aligned with updated principal axis mapping.")
        self._refresh_views()

    @QtCore.Slot(str)
    def on_view_axis(self, which: str) -> None:
        """Set camera to look along a named axis direction (applies to left view, syncs if linked)."""
        which = (which or "").strip().upper()
        vec_map = {
            "+X": (1.0, 0.0, 0.0),
            "-X": (-1.0, 0.0, 0.0),
            "+Y": (0.0, 1.0, 0.0),
            "-Y": (0.0, -1.0, 0.0),
            "+Z": (0.0, 0.0, 1.0),
            "-Z": (0.0, 0.0, -1.0),
        }
        up_map = {
            "+X": (0.0, 0.0, 1.0),
            "-X": (0.0, 0.0, 1.0),
            "+Y": (0.0, 0.0, 1.0),
            "-Y": (0.0, 0.0, 1.0),
            "+Z": (0.0, 1.0, 0.0),
            "-Z": (0.0, 1.0, 0.0),
        }
        if which not in vec_map:
            return

        self.view_left.view_vector(vec_map[which], viewup=up_map[which])
        self.view_left.render()
        if self.style.link_views:
            self._copy_camera(self.view_left, self.view_right)

    @QtCore.Slot()
    def on_isometric_view(self) -> None:
        """Set camera to isometric view (diagonal from corner)."""
        # Normalized isometric direction: looking from (+1, +1, +1) corner
        iso_vec = (1.0, 1.0, 1.0)
        iso_up = (0.0, 0.0, 1.0)
        self.view_left.view_vector(iso_vec, viewup=iso_up)
        self.view_left.render()
        if self.style.link_views:
            self._copy_camera(self.view_left, self.view_right)

    @QtCore.Slot(str)
    def on_rotate_axis(self, axis: str) -> None:
        """Rotate the aligned mesh 180° around given axis using core.rotate_alignment_result."""
        if self._result is None:
            QtWidgets.QMessageBox.information(self, "No mesh", "Load and align a mesh first.")
            return
        try:
            new_res = rotate_alignment_result(self._result, axis, degrees=180.0)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rotate failed", str(e))
            return

        self._result = new_res
        self.txt_axes.setPlainText(np.array2string(new_res.axes_matrix, precision=4, suppress_small=True))
        self.status.setText(f"Rotated 180° around {axis}-axis.")
        self._refresh_views()

    def closeEvent(self, event) -> None:
        try:
            self.view_left.close()
        except Exception:
            pass
        try:
            self.view_right.close()
        except Exception:
            pass
        super().closeEvent(event)


def _slider_int(min_v: int, max_v: int, init_v: int) -> QtWidgets.QSlider:
    s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    s.setMinimum(min_v)
    s.setMaximum(max_v)
    s.setValue(init_v)
    s.setSingleStep(1)
    return s


def _hline() -> QtWidgets.QFrame:
    ln = QtWidgets.QFrame()
    ln.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    ln.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return ln


def _labeled_row(label: str, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    v.addWidget(QtWidgets.QLabel(label))
    v.addLayout(layout)
    return w


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
