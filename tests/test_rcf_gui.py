import json
import io
import shutil
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


def _save_rgb_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _build_synthetic_scan(path: Path) -> None:
    canvas = np.full((220, 320, 3), 245, dtype=np.uint8)
    canvas[40:120, 30:120, 0] = 110
    canvas[40:120, 30:120, 1:] = 55
    canvas[60:170, 180:280, 0] = 135
    canvas[60:170, 180:280, 1:] = 70
    _save_rgb_image(path, canvas)


def _build_three_patch_scan(path: Path) -> None:
    canvas = np.full((240, 360, 3), 245, dtype=np.uint8)
    canvas[30:110, 20:110, 0] = 110
    canvas[30:110, 20:110, 1:] = 55
    canvas[70:180, 200:300, 0] = 135
    canvas[70:180, 200:300, 1:] = 70
    canvas[140:220, 30:120, 0] = 120
    canvas[140:220, 30:120, 1:] = 60
    _save_rgb_image(path, canvas)


def _build_rotated_scan(path: Path) -> None:
    canvas = Image.new("RGB", (420, 300), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(180, 50), (300, 90), (250, 240), (130, 200)], fill=(215, 218, 200), outline=(100, 110, 90))
    canvas.save(path)


def _build_background(path: Path, red_level: int) -> None:
    background = np.full((80, 80, 3), red_level, dtype=np.uint8)
    _save_rgb_image(path, background)


def _write_config(path: Path, film_path: Path, scanner_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "film_type: TEST",
                "background:",
                f"  film_path: {film_path.as_posix()}",
                f"  scanner_path: {scanner_path.as_posix()}",
                "segmentation:",
                "  min_area: 1000",
                "  padding: 4",
                "calibration:",
                "  background_quantile: 95",
                "  film_models:",
                "    TEST:",
                "      kind: polynomial",
                "      coefficients: [0.0, 100.0]",
            ]
        ),
        encoding="utf-8",
    )


def _write_multi_film_config(path: Path, hd_film_path: Path, ebt_film_path: Path, scanner_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "film_type: EBT3",
                "material_film_types:",
                "  HD: HDV2",
                "  EBT: EBT3",
                "background:",
                f"  film_path: {ebt_film_path.as_posix()}",
                f"  scanner_path: {scanner_path.as_posix()}",
                "  film_paths:",
                f"    HDV2: {hd_film_path.as_posix()}",
                f"    EBT3: {ebt_film_path.as_posix()}",
                "segmentation:",
                "  min_area: 1000",
                "  padding: 4",
                "calibration:",
                "  background_quantile: 95",
                "  film_models:",
                "    HDV2:",
                "      kind: polynomial",
                "      coefficients: [0.0, 120.0]",
                "    EBT3:",
                "      kind: polynomial",
                "      coefficients: [0.0, 60.0]",
            ]
        ),
        encoding="utf-8",
    )


def _write_stack_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "material_name": "HD",
                        "thickness": "105",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 0, "table_ID": 1, "Cutoff_ene": 3.6},
                    },
                    {
                        "material_name": "EBT",
                        "thickness": "280",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 1, "table_ID": 3, "Cutoff_ene": 28.6},
                    },
                    {
                        "material_name": "EBT",
                        "thickness": "280",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 2, "table_ID": 5, "Cutoff_ene": 43.7},
                    },
                    {
                        "material_name": "EBT",
                        "thickness": "280",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 3, "table_ID": 7, "Cutoff_ene": 56.0},
                    },
                    {
                        "material_name": "EBT",
                        "thickness": "280",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 4, "table_ID": 9, "Cutoff_ene": 66.6},
                    },
                ],
                "custom_materials": {},
            }
        ),
        encoding="utf-8",
    )


def _write_repo_style_config(root: Path, film_path: Path, scanner_path: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "rcf.example.yaml"
    _write_config(config_path, film_path, scanner_path)
    return config_path


def _build_app():
    from detectorclaw.rcf.gui import create_app

    return create_app()


def test_gui_patch_image_matches_autocrop_four_point_transform(tmp_path: Path) -> None:
    from detectorclaw.rcf.autocrop import detect_rcf_rectangles, four_point_transform

    scan_path = tmp_path / "rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]
    image_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/image")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

    gui_patch = np.array(Image.open(io.BytesIO(image_response.content)).convert("RGB"))
    source_bgr = cv2.imread(str(scan_path), cv2.IMREAD_COLOR)
    candidates, _, _ = detect_rcf_rectangles(
        image_bgr=source_bgr,
        expected_count=None,
        min_area_ratio=0.01,
        max_area_ratio=0.50,
        min_side_px=70,
        max_aspect_ratio=1.8,
        blur_ksize=5,
        morph_ksize=9,
        manual_threshold=None,
    )
    expected_bgr = four_point_transform(source_bgr, np.array(candidates[0]["box"], dtype=np.float32))
    expected_rgb = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)

    assert gui_patch.shape == expected_rgb.shape
    assert np.mean(np.abs(gui_patch.astype(np.int16) - expected_rgb.astype(np.int16))) < 1.0


def test_gui_loads_multiscan_session_and_updates_patch_state(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "RCF001.tif"
    scan_2_path = tmp_path / "RCF001_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    stack_config_path = tmp_path / "RCF1.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    config_path = _write_repo_style_config(tmp_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())

    page = client.get("/rcf/gui")
    assert page.status_code == 200
    assert "RCF 在线处理台" in page.text
    assert "校正后胶片" in page.text
    assert "发次编号" in page.text
    assert "恢复当前结果" in page.text
    assert "重新检测本发" in page.text
    assert "当前扫描胶片" in page.text
    assert "显示几何叠加" in page.text
    assert "步骤 1：修切割与旋转" in page.text
    assert "步骤 2：标记片序与重复片" in page.text
    assert "直接输入片序" in page.text
    assert "输入片序" in page.text
    assert "设为这个片序" in page.text
    assert "当前人工片序" in page.text
    assert "标记重复胶片" in page.text
    assert "自动切片校正" in page.text
    assert "中心 X" in page.text
    assert "精裁宽" in page.text
    assert "debugMoveSelectedPatch" in page.text
    assert "debugRotateSelectedPatch" in page.text
    assert "debugResizeSelectedPatch" in page.text
    assert "Apply Bounding Box" not in page.text
    assert "应用几何修改" in page.text

    response = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(tmp_path),
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["scan_count"] == 2
    assert payload["patch_count"] == 5
    session_id = payload["session_id"]

    state_response = client.get(f"/api/rcf/session/{session_id}/state")
    assert state_response.status_code == 200
    state = state_response.json()
    assert state["shot_id"] == "001"
    assert state["revision"] == 0
    assert state["detection_mode"] == "autocrop"
    assert state["session_source"] == "new_detection"
    assert state["autosaved_at"] is not None
    assert len(state["scans"]) == 2
    assert len(state["patches"]) == 5
    first_patch_id = state["patches"][0]["patch_id"]
    assert state["patches"][0]["detection_source"] == "autocrop"
    assert state["patches"][0]["assignment_status"] == "unassigned"
    assert state["patches"][0]["assigned_order"] is None
    assert state["patches"][0]["stack_mapping"] is None
    assert "rotated_rect" in state["patches"][0]

    geometry = state["patches"][0]["rotated_rect"]
    geometry_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch_id}/geometry",
        json={
            "rotated_rect": {
                "cx": geometry["cx"] + 5.0,
                "cy": geometry["cy"] + 7.0,
                "width": geometry["width"] + 11.0,
                "height": geometry["height"] + 13.0,
                "angle_deg": geometry["angle_deg"] + 3.0,
            }
        },
    )
    assert geometry_response.status_code == 200
    assert geometry_response.json()["rotated_rect"]["angle_deg"] == geometry["angle_deg"] + 3.0
    assert geometry_response.json()["modified_revision"] == 1

    revised_state = client.get(f"/api/rcf/session/{session_id}/state").json()
    assert revised_state["revision"] == 1
    assert revised_state["last_modified_patch_id"] == first_patch_id

    edge_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch_id}/edge",
        json={"edge_points": [[0, 0], [10, 10]]},
    )
    assert edge_response.status_code == 200
    assert round(edge_response.json()["angle_deg"], 3) == 45.0
    assert edge_response.json()["angle_source"] == "manual"
    assert edge_response.json()["angle_confidence"] == 1.0
    assert "low_confidence_angle" not in edge_response.json()["status_flags"]

    angle_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch_id}/angle",
        json={"angle_deg": 12.5},
    )
    assert angle_response.status_code == 200
    assert angle_response.json()["angle_deg"] == 12.5
    assert angle_response.json()["angle_source"] == "manual"
    assert angle_response.json()["angle_confidence"] == 1.0
    assert angle_response.json()["rotated_rect"]["angle_deg"] == 12.5

    crop_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch_id}/crop",
        json={"crop_bbox": [5, 6, 70, 80]},
    )
    assert crop_response.status_code == 200
    assert crop_response.json()["crop_bbox"] == [5, 6, 70, 80]

    missing_bbox_route = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch_id}/bbox",
        json={"bbox": [40, 50, 120, 130]},
    )
    assert missing_bbox_route.status_code == 404


def test_gui_reorders_patches_and_exports_session_files(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "scan_1.tif"
    scan_2_path = tmp_path / "scan_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    stack_config_path = tmp_path / "stack.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())
    response = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_1_path), str(scan_2_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
            "stack_config_file": str(stack_config_path),
        },
    )
    session_id = response.json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    reversed_patch_ids = list(reversed([patch["patch_id"] for patch in state["patches"]]))

    reorder_response = client.post(
        f"/api/rcf/session/{session_id}/order",
        json={"patch_ids": reversed_patch_ids},
    )
    assert reorder_response.status_code == 200
    reordered = reorder_response.json()["patches"]
    assert reordered[0]["patch_id"] == reversed_patch_ids[0]
    assert reordered[0]["global_order"] is None
    assert reorder_response.json()["revision"] > 0

    export_response = client.post(f"/api/rcf/session/{session_id}/export")
    assert export_response.status_code == 200
    export_payload = export_response.json()

    session_file = Path(export_payload["session_file"])
    review_file = Path(export_payload["review_file"])
    assert session_file.exists()
    assert review_file.exists()

    session_json = json.loads(session_file.read_text(encoding="utf-8"))
    review_json = json.loads(review_file.read_text(encoding="utf-8"))

    assert session_json["session_id"] == session_id
    assert len(session_json["patches"]) == 5
    assert review_json["patches"][0]["patch_id"] == reversed_patch_ids[0]
    assert review_json["patches"][0]["global_order"] is None
    assert review_json["patches"][0]["stack_mapping"] is None
    assert "rotated_rect" in review_json["patches"][0]


def test_gui_autosaves_and_restores_saved_shot_session(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "RCF001.tif"
    scan_2_path = tmp_path / "RCF001_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    stack_config_path = tmp_path / "RCF1.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    config_path = _write_repo_style_config(tmp_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())

    load_response = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(tmp_path),
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert load_response.status_code == 200, load_response.text
    assert load_response.json()["session_source"] == "new_detection"

    session_id = load_response.json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    first_patch = state["patches"][0]
    original_rect = first_patch["rotated_rect"]
    session_file = output_dir / "gui_session.json"
    assert session_file.exists()

    modified_rect = {
        "cx": original_rect["cx"] + 17.0,
        "cy": original_rect["cy"] + 9.0,
        "width": original_rect["width"] + 5.0,
        "height": original_rect["height"] + 7.0,
        "angle_deg": original_rect["angle_deg"] + 4.0,
    }
    update_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch['patch_id']}/geometry",
        json={"rotated_rect": modified_rect},
    )
    assert update_response.status_code == 200, update_response.text

    saved_session = json.loads(session_file.read_text(encoding="utf-8"))
    saved_patch = next(patch for patch in saved_session["patches"] if patch["patch_id"] == first_patch["patch_id"])
    assert saved_patch["rotated_rect"] == modified_rect
    assert saved_session["session_source"] == "new_detection"
    assert saved_session["autosaved_at"] is not None

    restored_response = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(tmp_path),
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert restored_response.status_code == 200, restored_response.text
    assert restored_response.json()["session_source"] == "restored"

    restored_state = client.get(f"/api/rcf/session/{restored_response.json()['session_id']}/state").json()
    restored_patch = next(patch for patch in restored_state["patches"] if patch["patch_id"] == first_patch["patch_id"])
    assert restored_state["session_source"] == "restored"
    assert restored_state["autosaved_at"] is not None
    assert restored_patch["rotated_rect"] == modified_rect
    assert restored_state["revision"] == 1
    assert restored_response.json()["session_id"] == session_id


def test_gui_restores_saved_shot_session_and_rebinds_scan_paths_after_reference_reorg(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy_root"
    current_root = tmp_path / "current_root"
    legacy_root.mkdir()
    current_root.mkdir()

    legacy_scan_1 = legacy_root / "RCF001.tif"
    legacy_scan_2 = legacy_root / "RCF001_2.tif"
    current_shot_dir = current_root / "reference" / "shots" / "shot_001"
    current_shot_dir.mkdir(parents=True)
    current_scan_1 = current_shot_dir / "RCF001.tif"
    current_scan_2 = current_shot_dir / "RCF001_2.tif"
    current_stack = current_shot_dir / "RCF1.json"
    film_background_path = current_root / "film_background.tif"
    scanner_background_path = current_root / "scanner_background.tif"
    output_dir = current_root / "runs" / "gui" / "shot_001_review"

    _build_synthetic_scan(legacy_scan_1)
    _build_three_patch_scan(legacy_scan_2)
    shutil.copy2(legacy_scan_1, current_scan_1)
    shutil.copy2(legacy_scan_2, current_scan_2)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    config_path = _write_repo_style_config(current_root, film_background_path, scanner_background_path)
    _write_stack_config(current_stack)

    client = TestClient(_build_app())
    first_load = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(current_root),
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert first_load.status_code == 200, first_load.text

    session_file = output_dir / "gui_session.json"
    saved_session = json.loads(session_file.read_text(encoding="utf-8"))
    saved_session["input_files"] = [str(current_scan_1), str(current_scan_2)]
    for version in saved_session["versions"]:
        for scan in version["scans"]:
            scan_index = int(scan["scan_index"])
            scan["scan_file"] = str(legacy_scan_1 if scan_index == 1 else legacy_scan_2)
        for patch in version["patches"]:
            scan_index = int(patch["scan_index"])
            patch["scan_file"] = str(legacy_scan_1 if scan_index == 1 else legacy_scan_2)
    session_file.write_text(json.dumps(saved_session, ensure_ascii=False, indent=2), encoding="utf-8")

    legacy_scan_1.unlink()
    legacy_scan_2.unlink()

    restored_response = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(current_root),
            "output_dir": str(output_dir),
        },
    )
    assert restored_response.status_code == 200, restored_response.text
    assert restored_response.json()["session_source"] == "restored"

    restored_state = client.get(f"/api/rcf/session/{restored_response.json()['session_id']}/state").json()
    assert [scan["scan_file"] for scan in restored_state["scans"]] == [str(current_scan_1), str(current_scan_2)]
    assert {patch["scan_file"] for patch in restored_state["patches"]} == {str(current_scan_1), str(current_scan_2)}

    preview_response = client.get(
        f"/api/rcf/session/{restored_response.json()['session_id']}/scan/1/image?max_dim=640&format=jpeg&quality=80"
    )
    assert preview_response.status_code == 200
    assert preview_response.headers["content-type"] == "image/jpeg"


def test_gui_state_exposes_version_metadata_and_redetect_creates_new_version(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "RCF001.tif"
    scan_2_path = tmp_path / "RCF001_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    stack_config_path = tmp_path / "RCF1.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    config_path = _write_repo_style_config(tmp_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())
    first_load = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(tmp_path),
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert first_load.status_code == 200, first_load.text
    first_session_id = first_load.json()["session_id"]
    first_version_id = first_load.json()["version_id"]

    first_state = client.get(f"/api/rcf/session/{first_session_id}/state").json()
    assert first_state["version_id"] == first_version_id
    assert first_state["active_version_id"] == first_version_id
    assert len(first_state["available_versions"]) == 1
    assert first_state["available_versions"][0]["version_id"] == first_version_id

    redetect_response = client.post(f"/api/rcf/session/{first_session_id}/redetect")
    assert redetect_response.status_code == 200, redetect_response.text
    assert redetect_response.json()["session_id"] == first_session_id
    assert redetect_response.json()["version_id"] != first_version_id

    redetected_state = client.get(f"/api/rcf/session/{first_session_id}/state").json()
    assert redetected_state["session_id"] == first_session_id
    assert redetected_state["version_id"] == redetect_response.json()["version_id"]
    assert redetected_state["active_version_id"] == redetect_response.json()["version_id"]
    assert redetected_state["revision"] == 0
    assert len(redetected_state["available_versions"]) == 2
    assert {item["version_id"] for item in redetected_state["available_versions"]} == {
        first_version_id,
        redetect_response.json()["version_id"],
    }


def test_gui_can_force_redetect_existing_shot_session(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "RCF001.tif"
    scan_2_path = tmp_path / "RCF001_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    stack_config_path = tmp_path / "RCF1.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    config_path = _write_repo_style_config(tmp_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())
    first_load = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(tmp_path),
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert first_load.status_code == 200, first_load.text
    session_id = first_load.json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    first_patch = state["patches"][0]
    modified_rect = dict(first_patch["rotated_rect"])
    modified_rect["cx"] += 23.0

    update_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch['patch_id']}/geometry",
        json={"rotated_rect": modified_rect},
    )
    assert update_response.status_code == 200, update_response.text

    redetect_response = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(tmp_path),
            "config_file": str(config_path),
            "output_dir": str(output_dir),
            "force_redetect": True,
        },
    )
    assert redetect_response.status_code == 200, redetect_response.text
    assert redetect_response.json()["session_source"] == "new_detection"

    redetected_state = client.get(f"/api/rcf/session/{redetect_response.json()['session_id']}/state").json()
    redetected_patch = next(patch for patch in redetected_state["patches"] if patch["patch_id"] == first_patch["patch_id"])
    assert redetected_state["session_source"] == "new_detection"
    assert redetected_state["revision"] == 0
    assert redetected_patch["rotated_rect"] != modified_rect


def test_gui_assigns_partial_stack_positions_and_exports_review_metadata(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "scan_1.tif"
    scan_2_path = tmp_path / "scan_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    stack_config_path = tmp_path / "stack.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_1_path), str(scan_2_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
            "stack_config_file": str(stack_config_path),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_ids = [patch["patch_id"] for patch in state["patches"]]

    assign_first = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_ids[0]}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 1},
    )
    assert assign_first.status_code == 200
    assert assign_first.json()["assignment_status"] == "assigned"
    assert assign_first.json()["assigned_order"] == 1
    assert assign_first.json()["process_in_current_batch"] is True
    assert assign_first.json()["stack_mapping"]["rcf_id"] == 0

    assign_second = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_ids[1]}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 1},
    )
    assert assign_second.status_code == 200
    assert assign_second.json()["assigned_order"] == 1
    assert assign_second.json()["stack_mapping"]["rcf_id"] == 0

    ignore_third = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_ids[2]}/assignment",
        json={"assignment_status": "ignored_duplicate"},
    )
    assert ignore_third.status_code == 200
    assert ignore_third.json()["assignment_status"] == "ignored_duplicate"
    assert ignore_third.json()["assigned_order"] is None
    assert ignore_third.json()["stack_mapping"] is None
    assert ignore_third.json()["process_in_current_batch"] is False

    exported = client.post(f"/api/rcf/session/{session_id}/export")
    assert exported.status_code == 200
    review_json = json.loads(Path(exported.json()["review_file"]).read_text(encoding="utf-8"))

    assert review_json["assigned_patch_count"] == 1
    assert review_json["processable_patch_count"] == 1
    assert review_json["ignored_patch_count"] == 1

    patch_map = {patch["patch_id"]: patch for patch in review_json["patches"]}
    assert patch_map[patch_ids[1]]["assigned_order"] == 1
    assert patch_map[patch_ids[1]]["stack_mapping"]["rcf_id"] == 0
    assert patch_map[patch_ids[0]]["assignment_status"] == "unassigned"
    assert patch_map[patch_ids[0]]["assigned_order"] is None
    assert patch_map[patch_ids[0]]["stack_mapping"] is None
    assert patch_map[patch_ids[2]]["assignment_status"] == "ignored_duplicate"
    assert patch_map[patch_ids[2]]["process_in_current_batch"] is False


def test_gui_keeps_existing_orders_after_clearing_assignment(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "scan_1.tif"
    scan_2_path = tmp_path / "scan_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    stack_config_path = tmp_path / "stack.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_1_path), str(scan_2_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
            "stack_config_file": str(stack_config_path),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_ids = [patch["patch_id"] for patch in state["patches"][:3]]

    for order, patch_id in enumerate(patch_ids, start=1):
        response = client.post(
            f"/api/rcf/session/{session_id}/patch/{patch_id}/assignment",
            json={"assignment_status": "assigned", "assigned_order": order},
        )
        assert response.status_code == 200

    clear_middle = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_ids[1]}/assignment",
        json={"assignment_status": "unassigned"},
    )
    assert clear_middle.status_code == 200
    assert clear_middle.json()["assignment_status"] == "unassigned"
    assert clear_middle.json()["assigned_order"] is None

    state_after = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_map = {patch["patch_id"]: patch for patch in state_after["patches"]}
    assert patch_map[patch_ids[0]]["assigned_order"] == 1
    assert patch_map[patch_ids[2]]["assigned_order"] == 3
    assert patch_map[patch_ids[2]]["stack_mapping"]["rcf_id"] == 2
    assert state_after["revision"] > 0


def test_gui_next_assignment_uses_smallest_missing_order(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "scan_1.tif"
    scan_2_path = tmp_path / "scan_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    stack_config_path = tmp_path / "stack.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_1_path), str(scan_2_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
            "stack_config_file": str(stack_config_path),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_ids = [patch["patch_id"] for patch in state["patches"][:3]]

    for order, patch_id in ((1, patch_ids[0]), (3, patch_ids[1])):
        response = client.post(
            f"/api/rcf/session/{session_id}/patch/{patch_id}/assignment",
            json={"assignment_status": "assigned", "assigned_order": order},
        )
        assert response.status_code == 200

    assign_missing = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_ids[2]}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 2},
    )
    assert assign_missing.status_code == 200
    assert assign_missing.json()["assigned_order"] == 2

    state_after = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_map = {patch["patch_id"]: patch for patch in state_after["patches"]}
    assert patch_map[patch_ids[0]]["assigned_order"] == 1
    assert patch_map[patch_ids[2]]["assigned_order"] == 2
    assert patch_map[patch_ids[1]]["assigned_order"] == 3


def test_gui_serves_browser_decodable_scan_preview(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    response = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]

    image_response = client.get(f"/api/rcf/session/{session_id}/scan/1/image")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

    decoded = Image.open(io.BytesIO(image_response.content))
    assert decoded.format == "PNG"
    assert decoded.size == (320, 220)


def test_gui_downsamples_large_scan_preview_for_browser(tmp_path: Path) -> None:
    scan_path = tmp_path / "large_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    large = np.full((1800, 3200, 3), 245, dtype=np.uint8)
    large[200:900, 200:1000, 0] = 110
    large[200:900, 200:1000, 1:] = 55
    _save_rgb_image(scan_path, large)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    response = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]

    image_response = client.get(f"/api/rcf/session/{session_id}/scan/1/image")
    assert image_response.status_code == 200
    decoded = Image.open(io.BytesIO(image_response.content))

    assert max(decoded.size) <= 1600
    assert decoded.size == (1600, 900)


def test_gui_serves_downsampled_jpeg_patch_preview(tmp_path: Path) -> None:
    scan_path = tmp_path / "large_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    large = np.full((1800, 3200, 3), 245, dtype=np.uint8)
    large[200:900, 200:1000, 0] = 110
    large[200:900, 200:1000, 1:] = 55
    _save_rgb_image(scan_path, large)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    response = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    image_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/image?max_dim=240&format=jpeg&quality=70")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    decoded = Image.open(io.BytesIO(image_response.content))

    assert max(decoded.size) <= 240
    assert decoded.format == "JPEG"


def test_gui_patch_preview_applies_saved_crop_bbox(tmp_path: Path) -> None:
    scan_path = tmp_path / "rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    original_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/image?max_dim=1000&format=png")
    assert original_response.status_code == 200
    original_image = Image.open(io.BytesIO(original_response.content))

    crop_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/crop",
        json={"crop_bbox": [10, 12, 90, 70]},
    )
    assert crop_response.status_code == 200

    cropped_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/image?max_dim=1000&format=png")
    assert cropped_response.status_code == 200
    cropped_image = Image.open(io.BytesIO(cropped_response.content))

    assert cropped_image.size == (90, 70)
    assert cropped_image.size[0] < original_image.size[0]
    assert cropped_image.size[1] < original_image.size[1]


def test_gui_crop_route_accepts_null_to_clear_saved_crop_bbox(tmp_path: Path) -> None:
    scan_path = tmp_path / "rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    original_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/image?max_dim=1000&format=png")
    original_image = Image.open(io.BytesIO(original_response.content))

    crop_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/crop",
        json={"crop_bbox": [10, 12, 90, 70]},
    )
    assert crop_response.status_code == 200
    assert crop_response.json()["crop_bbox"] == [10, 12, 90, 70]

    clear_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/crop",
        json={"crop_bbox": None},
    )
    assert clear_response.status_code == 200
    assert clear_response.json()["crop_bbox"] is None

    restored_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/image?max_dim=1000&format=png")
    restored_image = Image.open(io.BytesIO(restored_response.content))
    assert restored_image.size == original_image.size


def test_gui_serves_raw_patch_preview_from_server(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch = state["patches"][0]

    image_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch['patch_id']}/raw-image?max_dim=1000&format=jpeg&quality=70")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"

    decoded = Image.open(io.BytesIO(image_response.content))
    assert decoded.format == "JPEG"
    assert decoded.size == (patch["display_bbox"][2], patch["display_bbox"][3])


def test_gui_page_exposes_workflow_and_expert_view_modes() -> None:
    client = TestClient(_build_app())
    page = client.get("/rcf/gui")

    assert page.status_code == 200
    assert "向导视图" in page.text
    assert "专家视图" in page.text
    assert "原始扫描图" in page.text
    assert "修正图" in page.text
    assert "剂量计算图" in page.text
    assert "单片伪色" in page.text
    assert "上一片" in page.text
    assert "下一片" in page.text
    assert "Gray" in page.text
    assert "Turbo" in page.text
    assert "Jet" in page.text
    assert "自适应色标" in page.text
    assert "display_label" in page.text
    assert "format=jpeg" in page.text
    assert "viewerHighResDelayMs" in page.text
    assert "prefetchedViewerAssets" in page.text
    assert "fetch(url)" in page.text


def test_gui_page_uses_main_tabs_and_removes_stack_overview() -> None:
    client = TestClient(_build_app())
    page = client.get("/rcf/gui")

    assert page.status_code == 200
    assert "原始扫描图" in page.text
    assert "quick-assign-order" in page.text
    assert "堆栈总览" not in page.text
    assert 'id="order-list"' not in page.text
    assert "renderOrderList()" not in page.text
    assert "开始框选" in page.text
    assert "清除框选" in page.text
    assert "返回总览" in page.text
    assert "先给这片分配片序后再手动框选" in page.text
    assert "viewer-crop-editor" in page.text


def test_gui_page_prefetches_ready_dose_assets_for_tab_switching() -> None:
    client = TestClient(_build_app())
    page = client.get("/rcf/gui")

    assert page.status_code == 200
    assert "/assets/manifest" in page.text
    assert "warmReadyDoseAssetsFromManifest" in page.text
    assert "prefetchImageAndHydrateObjectUrl" in page.text


def test_gui_page_guards_empty_state_access_in_workflow_helpers() -> None:
    client = TestClient(_build_app())
    page = client.get("/rcf/gui")

    assert page.status_code == 200
    assert "return state?.patches?.find" in page.text
    assert "return state?.scans?.find" in page.text


def test_gui_page_clears_loading_copy_after_viewer_image_is_applied() -> None:
    client = TestClient(_build_app())
    page = client.get("/rcf/gui")

    assert page.status_code == 200
    assert "viewerEmpty.textContent = ''" in page.text


def test_gui_page_keeps_grid_tiles_on_stable_asset_urls() -> None:
    client = TestClient(_build_app())
    page = client.get("/rcf/gui")

    assert page.status_code == 200
    assert "image.src = imageUrl;" in page.text
    assert "image.src = getCachedViewerObjectUrl(imageUrl) || imageUrl;" not in page.text


def test_gui_page_stops_polling_on_unload() -> None:
    client = TestClient(_build_app())
    page = client.get("/rcf/gui")

    assert page.status_code == 200
    assert "function stopPolling()" in page.text
    assert "stopPolling();" in page.text


def test_gui_serves_empty_favicon() -> None:
    client = TestClient(_build_app())
    response = client.get("/favicon.ico")

    assert response.status_code == 204


def test_gui_state_exposes_display_label_for_assigned_and_unassigned(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    first_patch_id = state["patches"][0]["patch_id"]
    second_patch_id = state["patches"][1]["patch_id"]

    assert state["patches"][0]["display_label"] == "候1"
    assert state["patches"][1]["display_label"] == "候2"

    assign_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{second_patch_id}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 5},
    )
    assert assign_response.status_code == 200

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_map = {patch["patch_id"]: patch for patch in state["patches"]}
    assert patch_map[first_patch_id]["display_label"] == "候1"
    assert patch_map[second_patch_id]["display_label"] == "第 5 片"


def test_gui_session_keeps_detected_initial_angle(tmp_path: Path) -> None:
    scan_path = tmp_path / "rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    assert len(state["patches"]) == 1
    assert abs(state["patches"][0]["angle_deg"]) >= 5.0
    assert state["patches"][0]["angle_source"] in {"autocrop_min_area_rect", "contour_rect", "hough", "regionprops_fallback"}
    assert state["patches"][0]["angle_confidence"] > 0.0
    assert state["patches"][0]["status_flags"] == []
    assert abs(state["patches"][0]["rotated_rect"]["angle_deg"]) >= 5.0


def test_gui_allows_explicit_segment_detection_mode(tmp_path: Path) -> None:
    scan_path = tmp_path / "rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
            "detection_mode": "segment",
        },
    ).json()["session_id"]

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    assert state["detection_mode"] == "segment"
    assert state["patches"][0]["detection_source"] == "segment"
    assert state["patches"][0]["angle_source"] in {"contour_rect", "hough", "regionprops_fallback"}


def test_gui_serves_gray_and_pseudocolor_dose_previews(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    gray_response = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=gray&max_dim=300&format=jpeg"
    )
    assert gray_response.status_code == 200
    assert gray_response.headers["content-type"] == "image/jpeg"
    gray_image = Image.open(io.BytesIO(gray_response.content))
    assert gray_image.mode == "RGB"

    pseudo_response = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=300&format=jpeg"
    )
    assert pseudo_response.status_code == 200
    assert pseudo_response.headers["content-type"] == "image/jpeg"
    pseudo_image = Image.open(io.BytesIO(pseudo_response.content))
    assert pseudo_image.mode == "RGB"
    assert pseudo_image.size == gray_image.size

    jet_response = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=jet&max_dim=300&format=jpeg"
    )
    assert jet_response.status_code == 200
    assert jet_response.headers["content-type"] == "image/jpeg"
    jet_image = Image.open(io.BytesIO(jet_response.content))
    assert jet_image.mode == "RGB"
    assert jet_image.size == gray_image.size


def test_gui_dose_export_supports_lossless_tiff(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    response = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-export?palette=turbo&max_dim=300&format=tiff"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    exported = Image.open(io.BytesIO(response.content))
    assert exported.format == "TIFF"


def test_gui_dose_preview_passes_max_dim_into_patch_extraction(tmp_path: Path) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    if hasattr(gui_module, "DOSE_ARRAY_CACHE"):
        gui_module.DOSE_ARRAY_CACHE.clear()

    original_extract = gui_module.preview.extract_patch_image
    seen: list[int | None] = []

    def wrapped(scan_file, quad_points, crop_bbox=None, backend="auto", max_dim=None):
        seen.append(max_dim)
        return original_extract(scan_file, quad_points, crop_bbox=crop_bbox, backend="cpu", max_dim=max_dim)

    gui_module.preview.extract_patch_image = wrapped
    try:
        response = client.get(
            f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=300&format=jpeg"
        )
    finally:
        gui_module.preview.extract_patch_image = original_extract

    assert response.status_code == 200
    assert 300 in seen


def test_gui_dose_preview_rejects_unknown_palette(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=viridis&max_dim=300")
    assert response.status_code == 400
    assert "dose palette must be gray, turbo, or jet" in response.json()["detail"]


def test_gui_dose_stats_endpoint_returns_adaptive_values(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["patch_id"] == patch_id
    assert payload["film_type"] == "TEST"
    assert payload["dose_min"] >= 0.0
    assert payload["dose_max"] >= payload["dose_min"]
    assert payload["dose_mean"] >= payload["dose_min"]


def test_gui_reuses_cached_dose_array_across_dose_views(tmp_path: Path) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(gui_module.create_app(disable_precompute=True))
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    original = gui_module.dose_from_patch
    calls = {"count": 0}

    def wrapped(*args, **kwargs):
        import inspect

        frame = inspect.currentframe()
        tracked = False
        while frame is not None:
            local_patch = frame.f_locals.get("patch")
            local_session = frame.f_locals.get("session")
            if isinstance(local_patch, dict) and isinstance(local_session, dict):
                if local_patch.get("patch_id") == patch_id and local_session.get("session_id") == session_id:
                    tracked = True
                    break
            frame = frame.f_back
        if tracked:
            calls["count"] += 1
        return original(*args, **kwargs)

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    if hasattr(gui_module, "DOSE_ARRAY_CACHE"):
        gui_module.DOSE_ARRAY_CACHE.clear()
    gui_module.dose_from_patch = wrapped
    try:
        baseline = calls["count"]
        assert client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=300").status_code == 200
        after_first = calls["count"]
        assert client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-stats").status_code == 200
        after_stats = calls["count"]
        assert client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=jet&max_dim=300").status_code == 200
        assert calls["count"] == after_stats
        assert after_first - baseline <= 2
    finally:
        gui_module.dose_from_patch = original


def test_gui_dose_overview_prewarm_batches_assigned_patches(tmp_path: Path) -> None:
    import detectorclaw.rcf.gui as gui_module
    import detectorclaw.rcf.preview as preview_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_ids = [patch["patch_id"] for patch in state["patches"]]

    for order, patch_id in enumerate(patch_ids[:2], start=1):
        response = client.post(
            f"/api/rcf/session/{session_id}/patch/{patch_id}/assignment",
            json={"assignment_status": "assigned", "assigned_order": order},
        )
        assert response.status_code == 200

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    if hasattr(gui_module, "DOSE_ARRAY_CACHE"):
        gui_module.DOSE_ARRAY_CACHE.clear()

    original = preview_module.extract_patch_images
    calls: list[int] = []

    def wrapped(scan_file, quad_points_list, crop_bboxes=None, backend="auto"):
        calls.append(len(quad_points_list))
        return original(scan_file, quad_points_list, crop_bboxes=crop_bboxes, backend=backend)

    preview_module.extract_patch_images = wrapped
    try:
        response = client.get(
            f"/api/rcf/session/{session_id}/dose-overview-prewarm?palette=turbo&max_dim=240&format=jpeg&quality=90"
        )
    finally:
        preview_module.extract_patch_images = original

    assert response.status_code == 200
    payload = response.json()
    assert payload["patch_count"] == 2
    assert calls in ([], [2])
    assert len(gui_module.DOSE_STATS_CACHE) >= 2
    assert sum(1 for key in gui_module.PREVIEW_CACHE if key and key[0] == "dose-patch") >= 2


def test_gui_load_starts_session_precompute_and_populates_caches(tmp_path: Path, monkeypatch) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    gui_module.DOSE_ARRAY_CACHE.clear()
    gui_module.DOSE_PREVIEW_ARRAY_CACHE.clear()

    original_load_dose_context = gui_module._load_dose_context

    def cpu_only_load_dose_context(session, patch=None):
        payload = original_load_dose_context(session, patch=patch)
        payload["backend"] = "cpu"
        return payload

    monkeypatch.setattr(gui_module, "_load_dose_context", cpu_only_load_dose_context)

    client = TestClient(gui_module.create_app())
    response = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    session_id = payload["session_id"]
    deadline = time.time() + 15.0
    state = None
    while time.time() < deadline:
        state = client.get(f"/api/rcf/session/{session_id}/state").json()
        if state["precompute_status"]["state"] == "done":
            break
        time.sleep(0.1)

    assert state is not None
    assert state["precompute_status"]["state"] == "done"
    assert state["precompute_status"]["warmed_count"] == state["precompute_status"]["total_count"]

    assert sum(1 for key in gui_module.PREVIEW_CACHE if key and key[0] == "scan") >= 1
    assert sum(1 for key in gui_module.PREVIEW_CACHE if key and key[0] == "patch") >= 1
    assert sum(1 for key in gui_module.PREVIEW_CACHE if key and key[0] == "raw-patch") >= 1
    assert sum(1 for key in gui_module.PREVIEW_CACHE if key and key[0] == "dose-patch") >= 1
    assert len(gui_module.DOSE_STATS_CACHE) >= 1


def test_gui_assignment_requeues_session_precompute_for_current_revision(tmp_path: Path, monkeypatch) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    gui_module.DOSE_ARRAY_CACHE.clear()
    gui_module.DOSE_PREVIEW_ARRAY_CACHE.clear()

    original_load_dose_context = gui_module._load_dose_context

    def cpu_only_load_dose_context(session, patch=None):
        payload = original_load_dose_context(session, patch=patch)
        payload["backend"] = "cpu"
        return payload

    monkeypatch.setattr(gui_module, "_load_dose_context", cpu_only_load_dose_context)

    client = TestClient(gui_module.create_app())
    payload = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()
    session_id = payload["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch = state["patches"][0]

    response = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch['patch_id']}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 1},
    )
    assert response.status_code == 200

    deadline = time.time() + 15.0
    updated_state = None
    while time.time() < deadline:
        updated_state = client.get(f"/api/rcf/session/{session_id}/state").json()
        status = updated_state["precompute_status"]
        if status["state"] == "done" and status["revision"] == updated_state["revision"]:
            break
        time.sleep(0.1)

    assert updated_state is not None
    updated_patch = next(item for item in updated_state["patches"] if item["patch_id"] == patch["patch_id"])
    assert updated_state["precompute_status"]["state"] == "done"
    assert updated_state["precompute_status"]["revision"] == updated_state["revision"]
    assert any(
        key
        for key in gui_module.PREVIEW_CACHE
        if key
        and key[0] == "dose-patch"
        and key[2] == updated_patch["patch_id"]
        and key[3] == updated_patch["modified_revision"]
    )


def test_gui_dose_image_cache_only_skips_synchronous_compute(tmp_path: Path) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    gui_module.DOSE_ARRAY_CACHE.clear()
    gui_module.DOSE_PREVIEW_ARRAY_CACHE.clear()

    client = TestClient(gui_module.create_app(disable_precompute=True))
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    miss = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=320&format=jpeg&cache_only=true"
    )
    assert miss.status_code == 425
    assert miss.json()["detail"] == "Dose preview cache miss"

    warm = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=320&format=jpeg"
    )
    assert warm.status_code == 200

    hit = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=320&format=jpeg&cache_only=true"
    )
    assert hit.status_code == 200
    assert hit.headers["content-type"] == "image/jpeg"


def test_gui_assets_manifest_reports_assigned_patch_readiness(tmp_path: Path) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    gui_module.DOSE_ARRAY_CACHE.clear()
    gui_module.DOSE_PREVIEW_ARRAY_CACHE.clear()

    client = TestClient(gui_module.create_app(disable_precompute=True))
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]
    assign = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 1},
    )
    assert assign.status_code == 200

    manifest_before = client.get(f"/api/rcf/session/{session_id}/assets/manifest")
    assert manifest_before.status_code == 200
    payload_before = manifest_before.json()
    assert payload_before["assigned_patch_count"] == 1
    patch_payload = payload_before["patches"][0]
    assert patch_payload["display_label"] == "第 1 片"
    variant_before = next(item for item in patch_payload["variants"] if item["variant_id"] == "dose_single_320_turbo")
    assert variant_before["ready"] is False
    assert variant_before["asset_id"] is not None

    warm = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=320&format=jpeg"
    )
    assert warm.status_code == 200

    manifest_after = client.get(f"/api/rcf/session/{session_id}/assets/manifest")
    assert manifest_after.status_code == 200
    payload_after = manifest_after.json()
    patch_after = payload_after["patches"][0]
    variant_after = next(item for item in patch_after["variants"] if item["variant_id"] == "dose_single_320_turbo")
    assert variant_after["ready"] is True
    asset = client.get(f"/api/rcf/session/{session_id}/assets/{variant_after['asset_id']}")
    assert asset.status_code == 200
    assert asset.headers["content-type"] == "image/jpeg"


def test_gui_precompute_status_and_start_endpoints(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]

    status = client.get(f"/api/rcf/session/{session_id}/precompute/status")
    assert status.status_code == 200
    assert status.json()["state"] in {"idle", "queued", "running", "done", "error", "superseded"}
    assert "interactive_queue" in status.json()
    assert "bulk_queue" in status.json()
    assert "inflight_batch" in status.json()
    assert "backend" in status.json()

    start = client.post(f"/api/rcf/session/{session_id}/precompute/start")
    assert start.status_code == 200
    assert start.json()["state"] in {"queued", "running", "done", "error", "superseded"}


def test_gui_dose_asset_disk_cache_survives_memory_clear(tmp_path: Path) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    gui_module.DOSE_ARRAY_CACHE.clear()
    gui_module.DOSE_PREVIEW_ARRAY_CACHE.clear()

    client = TestClient(gui_module.create_app(disable_precompute=True))
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    first = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=320&format=jpeg"
    )
    assert first.status_code == 200

    gui_module.PREVIEW_CACHE.clear()
    second = client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=320&format=jpeg&cache_only=true"
    )
    assert second.status_code == 200
    assert second.content == first.content


def test_gui_assets_manifest_reads_disk_cache_readiness(tmp_path: Path) -> None:
    import detectorclaw.rcf.gui as gui_module

    scan_path = tmp_path / "scan_1.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    gui_module.PREVIEW_CACHE.clear()
    gui_module.DOSE_STATS_CACHE.clear()
    gui_module.DOSE_ARRAY_CACHE.clear()
    gui_module.DOSE_PREVIEW_ARRAY_CACHE.clear()

    client = TestClient(gui_module.create_app(disable_precompute=True))
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]
    assert client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 1},
    ).status_code == 200
    assert client.get(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=320&format=jpeg"
    ).status_code == 200

    gui_module.PREVIEW_CACHE.clear()
    manifest = client.get(f"/api/rcf/session/{session_id}/assets/manifest")
    assert manifest.status_code == 200
    payload = manifest.json()
    patch_payload = payload["patches"][0]
    variant = next(item for item in patch_payload["variants"] if item["variant_id"] == "dose_single_320_turbo")
    assert variant["ready"] is True
def test_gui_dose_preview_uses_material_specific_backgrounds_and_models(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    hd_background_path = tmp_path / "hd_background.tif"
    ebt_background_path = tmp_path / "ebt_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    stack_config_path = tmp_path / "RCF1.json"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    _build_background(hd_background_path, red_level=220)
    _build_background(ebt_background_path, red_level=170)
    _build_background(scanner_background_path, red_level=10)
    _write_multi_film_config(config_path, hd_background_path, ebt_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
            "stack_config_file": str(stack_config_path),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    first_patch_id = state["patches"][0]["patch_id"]
    second_patch_id = state["patches"][1]["patch_id"]

    assign_hd = client.post(
        f"/api/rcf/session/{session_id}/patch/{first_patch_id}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 1},
    )
    assert assign_hd.status_code == 200
    assign_ebt = client.post(
        f"/api/rcf/session/{session_id}/patch/{second_patch_id}/assignment",
        json={"assignment_status": "assigned", "assigned_order": 2},
    )
    assert assign_ebt.status_code == 200

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    hd_patch = next(patch for patch in state["patches"] if patch["patch_id"] == first_patch_id)
    ebt_patch = next(patch for patch in state["patches"] if patch["patch_id"] == second_patch_id)
    assert hd_patch["stack"]["material_name"] == "HD"
    assert ebt_patch["stack"]["material_name"] == "EBT"

    from detectorclaw.rcf.gui import _load_dose_context

    session = client.app.state.store.get_session(session_id)
    hd_context = _load_dose_context(session, patch=hd_patch)
    ebt_context = _load_dose_context(session, patch=ebt_patch)
    assert hd_context["film_type"] == "HDV2"
    assert ebt_context["film_type"] == "EBT3"
    assert hd_context["film_background_mean"] != ebt_context["film_background_mean"]


def test_gui_state_reports_dose_unavailable_when_backgrounds_are_missing(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    config_path.write_text(
        "\n".join(
            [
                "film_type: TEST",
                "background:",
                "  film_path: C:/path/to/film_background.tif",
                "  scanner_path: C:/path/to/scanner_background.tif",
                "segmentation:",
                "  min_area: 1000",
                "  padding: 4",
                "calibration:",
                "  background_quantile: 95",
                "  film_models:",
                "    TEST:",
                "      kind: polynomial",
                "      coefficients: [0.0, 100.0]",
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    assert state["dose_available"] is False
    assert "Background file not found" in state["dose_error"]
    assert state["dose_config_source"] == "explicit"


def test_gui_dose_preview_returns_400_for_missing_backgrounds(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan_1.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_synthetic_scan(scan_path)
    config_path.write_text(
        "\n".join(
            [
                "film_type: TEST",
                "background:",
                "  film_path: C:/path/to/film_background.tif",
                "  scanner_path: C:/path/to/scanner_background.tif",
                "segmentation:",
                "  min_area: 1000",
                "  padding: 4",
                "calibration:",
                "  background_quantile: 95",
                "  film_models:",
                "    TEST:",
                "      kind: polynomial",
                "      coefficients: [0.0, 100.0]",
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=turbo&max_dim=300")
    assert response.status_code == 400
    assert "Background file not found" in response.json()["detail"]


def test_gui_shot_mode_prefers_discovered_real_config_over_example(tmp_path: Path) -> None:
    shot_dir = tmp_path / "reference" / "shots" / "shot_001"
    scan_1_path = shot_dir / "RCF001.tif"
    scan_2_path = shot_dir / "RCF001_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    stack_config_path = shot_dir / "RCF1.json"
    config_dir = tmp_path / "configs"
    real_config_path = config_dir / "rcf.lab.yaml"
    example_config_path = config_dir / "rcf.example.yaml"

    shot_dir.mkdir(parents=True, exist_ok=True)
    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_stack_config(stack_config_path)
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_config(real_config_path, film_background_path, scanner_background_path)
    example_config_path.write_text(
        "\n".join(
            [
                "film_type: TEST",
                "background:",
                "  film_path: C:/path/to/film_background.tif",
                "  scanner_path: C:/path/to/scanner_background.tif",
                "segmentation:",
                "  min_area: 1000",
                "  padding: 4",
                "calibration:",
                "  background_quantile: 95",
                "  film_models:",
                "    TEST:",
                "      kind: polynomial",
                "      coefficients: [0.0, 100.0]",
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "shot_id": "001",
            "data_root": str(tmp_path),
        },
    ).json()["session_id"]

    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    assert state["dose_available"] is True
    assert state["dose_config_source"] == "discovered"
    assert state["config_file"] == str(real_config_path)
    assert state["input_files"] == [str(scan_1_path), str(scan_2_path)]
    assert state["output_dir"] == str(tmp_path / "runs" / "gui" / "shot_001_review")


def test_gui_dose_preview_respects_saved_crop_bbox(tmp_path: Path) -> None:
    scan_path = tmp_path / "rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "gui_out"

    _build_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    client = TestClient(_build_app())
    session_id = client.post(
        "/api/rcf/session/load",
        json={
            "input_files": [str(scan_path)],
            "config_file": str(config_path),
            "output_dir": str(output_dir),
        },
    ).json()["session_id"]
    state = client.get(f"/api/rcf/session/{session_id}/state").json()
    patch_id = state["patches"][0]["patch_id"]

    original_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=gray&max_dim=1000")
    assert original_response.status_code == 200
    original_image = Image.open(io.BytesIO(original_response.content))

    crop_response = client.post(
        f"/api/rcf/session/{session_id}/patch/{patch_id}/crop",
        json={"crop_bbox": [10, 12, 90, 70]},
    )
    assert crop_response.status_code == 200

    cropped_response = client.get(f"/api/rcf/session/{session_id}/patch/{patch_id}/dose-image?palette=gray&max_dim=1000")
    assert cropped_response.status_code == 200
    cropped_image = Image.open(io.BytesIO(cropped_response.content))

    assert cropped_image.size == (90, 70)
    assert cropped_image.size[0] < original_image.size[0]
    assert cropped_image.size[1] < original_image.size[1]
