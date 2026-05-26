use anyhow::{Context, Result, anyhow};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::{ColorType, DynamicImage, GrayImage, ImageEncoder, ImageReader, Rgb, RgbImage};
use imageproc::geometric_transformations::{Interpolation, Projection, warp_into};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct ScanPreviewRequest {
    scan_file: String,
    max_dim: u32,
    preview_format: String,
    quality: u8,
}

#[derive(Debug, Deserialize)]
struct BBoxPreviewRequest {
    scan_file: String,
    bbox: [u32; 4],
    max_dim: u32,
    preview_format: String,
    quality: u8,
}

#[derive(Debug, Deserialize)]
struct PatchPreviewRequest {
    scan_file: String,
    quad_points: Vec<[f32; 2]>,
    crop_bbox: Option<[u32; 4]>,
    max_dim: u32,
    preview_format: String,
    quality: u8,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DoseFilmModel {
    Polynomial { coefficients: Vec<f64> },
    PowerLaw { scale: f64, exponent: f64 },
}

#[derive(Debug, Deserialize)]
struct DoseBatchTaskRequest {
    task_id: String,
    scan_file: String,
    quad_points: Vec<[f32; 2]>,
    crop_bbox: Option<[u32; 4]>,
    max_dim: u32,
    preview_format: String,
    quality: u8,
    palette: String,
    film_background_mean: f64,
    scanner_background_mean: f64,
    film_model: DoseFilmModel,
    background_quantile: f64,
}

#[derive(Debug, Deserialize)]
struct DoseBatchRequest {
    tasks: Vec<DoseBatchTaskRequest>,
}

#[derive(Debug, Serialize)]
struct DoseBatchTaskResponse {
    task_id: String,
    media_type: String,
    content_hex: String,
    dose_min: f64,
    dose_max: f64,
    dose_mean: f64,
}

#[derive(Debug, Serialize)]
struct DoseBatchResponse {
    results: Vec<DoseBatchTaskResponse>,
}

#[derive(Debug, Deserialize)]
struct SegmentDetectRequest {
    scan_file: String,
    min_area: u32,
    padding: u32,
    sort_mode: String,
}

#[derive(Debug, Serialize)]
struct PreviewResponse {
    media_type: String,
    content_hex: String,
}

#[derive(Debug, Clone, Serialize)]
struct SegmentPatch {
    order: u32,
    bbox: [u32; 4],
    angle_deg: f32,
    angle_confidence: f32,
    angle_source: String,
    status_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SegmentComponent {
    component_id: u32,
    area: u32,
    bbox: [u32; 4],
    angle_deg: f32,
    angle_confidence: f32,
    angle_source: String,
    status_flags: Vec<String>,
    kept: bool,
}

#[derive(Debug, Serialize)]
struct SegmentDetectResponse {
    component_count: u32,
    mask_png_hex: String,
    patches: Vec<SegmentPatch>,
    components: Vec<SegmentComponent>,
}

#[derive(Debug, Clone)]
struct GeometryEstimate {
    angle_deg: f32,
    angle_confidence: f32,
    angle_source: &'static str,
    status_flags: Vec<String>,
}

#[derive(Debug, Clone)]
struct Component {
    area: u32,
    min_x: usize,
    min_y: usize,
    mask: Vec<u8>,
    width: usize,
    height: usize,
}

pub fn segment_detect_json(input: &str) -> Result<String> {
    let request: SegmentDetectRequest =
        serde_json::from_str(input).context("failed parsing json request")?;
    let image = load_rgb_image(PathBuf::from(&request.scan_file))?;
    let response = segment_detect(&image, &request)?;
    serde_json::to_string(&response).context("failed serializing json response")
}

pub fn dose_batch_json(input: &str) -> Result<String> {
    let request: DoseBatchRequest =
        serde_json::from_str(input).context("failed parsing json request")?;
    let response = dose_batch(&request)?;
    serde_json::to_string(&response).context("failed serializing json response")
}

pub fn preview_command_json(command: &str, input: &str) -> Result<String> {
    match command {
        "scan-preview" => {
            let request: ScanPreviewRequest =
                serde_json::from_str(input).context("failed parsing json request")?;
            let image = load_rgb_image(PathBuf::from(request.scan_file))?;
            let resized = resize_for_preview(&image, request.max_dim);
            serde_json::to_string(&encode_image(
                &resized,
                &request.preview_format,
                request.quality,
            )?)
            .context("failed serializing json response")
        }
        "bbox-preview" => {
            let request: BBoxPreviewRequest =
                serde_json::from_str(input).context("failed parsing json request")?;
            let image = load_rgb_image(PathBuf::from(request.scan_file))?;
            let cropped = crop_image(&image, request.bbox);
            let resized = resize_for_preview(&cropped, request.max_dim);
            serde_json::to_string(&encode_image(
                &resized,
                &request.preview_format,
                request.quality,
            )?)
            .context("failed serializing json response")
        }
        "patch-preview" => {
            let request: PatchPreviewRequest =
                serde_json::from_str(input).context("failed parsing json request")?;
            if request.quad_points.len() != 4 {
                return Err(anyhow!("quad_points must contain four points"));
            }
            let image = load_rgb_image(PathBuf::from(request.scan_file))?;
            let warped = patch_preview(&image, &request.quad_points)?;
            let cropped = match request.crop_bbox {
                Some(bbox) => crop_image(&warped, bbox),
                None => warped,
            };
            let resized = resize_for_preview(&cropped, request.max_dim);
            serde_json::to_string(&encode_image(
                &resized,
                &request.preview_format,
                request.quality,
            )?)
            .context("failed serializing json response")
        }
        "dose-batch" => dose_batch_json(input),
        other => Err(anyhow!("unsupported preview command: {other}")),
    }
}

fn main() -> Result<()> {
    let command = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("missing command"))?;
    match command.as_str() {
        "scan-preview" => {
            let request_text = read_request_text()?;
            let response = preview_command_json("scan-preview", &request_text)?;
            write_text(&response)?;
        }
        "bbox-preview" => {
            let request_text = read_request_text()?;
            let response = preview_command_json("bbox-preview", &request_text)?;
            write_text(&response)?;
        }
        "patch-preview" => {
            let request_text = read_request_text()?;
            let response = preview_command_json("patch-preview", &request_text)?;
            write_text(&response)?;
        }
        "dose-batch" => {
            let request_text = read_request_text()?;
            let response = preview_command_json("dose-batch", &request_text)?;
            write_text(&response)?;
        }
        "segment-detect" => {
            let request: SegmentDetectRequest = read_request()?;
            let image = load_rgb_image(PathBuf::from(&request.scan_file))?;
            let response = segment_detect(&image, &request)?;
            write_json(&response)?;
        }
        _ => return Err(anyhow!("unsupported command: {command}")),
    }
    Ok(())
}

fn read_request<T: for<'de> Deserialize<'de>>() -> Result<T> {
    let input = read_request_text()?;
    serde_json::from_str(&input).context("failed parsing json request")
}

fn read_request_text() -> Result<String> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .context("failed reading stdin")?;
    Ok(input)
}

fn write_json<T: Serialize>(value: &T) -> Result<()> {
    let encoded = serde_json::to_string(value)?;
    std::io::stdout()
        .write_all(encoded.as_bytes())
        .context("failed writing response")
}

fn write_text(value: &str) -> Result<()> {
    std::io::stdout()
        .write_all(value.as_bytes())
        .context("failed writing response")
}

fn load_rgb_image(path: PathBuf) -> Result<RgbImage> {
    Ok(ImageReader::open(path)?.decode()?.to_rgb8())
}

fn resize_for_preview(image: &RgbImage, max_dim: u32) -> RgbImage {
    if max_dim == 0 {
        return image.clone();
    }
    let (width, height) = image.dimensions();
    let largest = width.max(height);
    if largest <= max_dim {
        return image.clone();
    }
    let scale = max_dim as f32 / largest as f32;
    let new_width = ((width as f32) * scale).round().max(1.0) as u32;
    let new_height = ((height as f32) * scale).round().max(1.0) as u32;
    image::imageops::resize(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Lanczos3,
    )
}

fn crop_image(image: &RgbImage, bbox: [u32; 4]) -> RgbImage {
    let (image_width, image_height) = image.dimensions();
    if image_width == 0 || image_height == 0 {
        return image.clone();
    }
    let x = bbox[0].min(image_width.saturating_sub(1));
    let y = bbox[1].min(image_height.saturating_sub(1));
    let width = bbox[2].max(1).min(image_width.saturating_sub(x));
    let height = bbox[3].max(1).min(image_height.saturating_sub(y));
    image::imageops::crop_imm(image, x, y, width.max(1), height.max(1)).to_image()
}

fn patch_preview(image: &RgbImage, quad_points: &[[f32; 2]]) -> Result<RgbImage> {
    let ordered = order_points(quad_points)?;
    let max_width = distance(ordered[2], ordered[3])
        .max(distance(ordered[1], ordered[0]))
        .round()
        .max(2.0) as u32;
    let max_height = distance(ordered[1], ordered[2])
        .max(distance(ordered[0], ordered[3]))
        .round()
        .max(2.0) as u32;

    let from = [
        (ordered[0][0], ordered[0][1]),
        (ordered[1][0], ordered[1][1]),
        (ordered[2][0], ordered[2][1]),
        (ordered[3][0], ordered[3][1]),
    ];
    let to = [
        (0.0, 0.0),
        ((max_width.saturating_sub(1)) as f32, 0.0),
        (
            (max_width.saturating_sub(1)) as f32,
            (max_height.saturating_sub(1)) as f32,
        ),
        (0.0, (max_height.saturating_sub(1)) as f32),
    ];
    let projection = Projection::from_control_points(from, to)
        .ok_or_else(|| anyhow!("failed to build perspective projection"))?;
    let mut output = RgbImage::from_pixel(max_width, max_height, Rgb([255, 255, 255]));
    warp_into(
        image,
        &projection,
        Interpolation::Bilinear,
        Rgb([255, 255, 255]),
        &mut output,
    );
    Ok(output)
}

fn encode_image(image: &RgbImage, preview_format: &str, quality: u8) -> Result<PreviewResponse> {
    let mut bytes = Vec::new();
    match preview_format {
        "jpeg" => {
            let encoder = JpegEncoder::new_with_quality(&mut bytes, quality.clamp(1, 95));
            let dyn_image = DynamicImage::ImageRgb8(image.clone());
            encoder.write_image(
                dyn_image.as_bytes(),
                dyn_image.width(),
                dyn_image.height(),
                ColorType::Rgb8.into(),
            )?;
            Ok(PreviewResponse {
                media_type: "image/jpeg".to_string(),
                content_hex: hex::encode(bytes),
            })
        }
        "png" => {
            let encoder = PngEncoder::new(&mut bytes);
            encoder.write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                ColorType::Rgb8.into(),
            )?;
            Ok(PreviewResponse {
                media_type: "image/png".to_string(),
                content_hex: hex::encode(bytes),
            })
        }
        other => Err(anyhow!("unsupported preview format: {other}")),
    }
}

fn dose_batch(request: &DoseBatchRequest) -> Result<DoseBatchResponse> {
    if request.tasks.is_empty() {
        return Ok(DoseBatchResponse { results: Vec::new() });
    }
    if request.tasks.len() > 8 {
        return Err(anyhow!("dose-batch supports at most 8 tasks per request"));
    }

    let mut image_cache: HashMap<String, Arc<RgbImage>> = HashMap::new();
    for task in &request.tasks {
        if task.quad_points.len() != 4 {
            return Err(anyhow!("quad_points must contain four points"));
        }
        if !image_cache.contains_key(&task.scan_file) {
            let image = load_rgb_image(PathBuf::from(&task.scan_file))
                .with_context(|| format!("failed loading scan file {}", task.scan_file))?;
            image_cache.insert(task.scan_file.clone(), Arc::new(image));
        }
    }
    let image_cache = Arc::new(image_cache);

    let ordered_results: Vec<Result<DoseBatchTaskResponse>> = request
        .tasks
        .par_iter()
        .map(|task| {
            let image = image_cache
                .get(&task.scan_file)
                .ok_or_else(|| anyhow!("scan image missing from cache"))?;
            process_dose_task(task, image)
        })
        .collect();

    let mut results = Vec::with_capacity(ordered_results.len());
    for item in ordered_results {
        results.push(item?);
    }
    Ok(DoseBatchResponse { results })
}

fn process_dose_task(task: &DoseBatchTaskRequest, image: &RgbImage) -> Result<DoseBatchTaskResponse> {
    let warped = patch_preview(image, &task.quad_points)?;
    let cropped = match task.crop_bbox {
        Some(bbox) => crop_image(&warped, bbox),
        None => warped,
    };
    let resized = resize_for_preview(&cropped, task.max_dim);
    let (dose_gray, dose_min, dose_max, dose_mean) = compute_dose_gray(
        &resized,
        task.film_background_mean,
        task.scanner_background_mean,
        &task.film_model,
        task.background_quantile,
    )?;
    let preview_rgb = colorize_dose_gray(&dose_gray, &task.palette)?;
    let encoded = encode_image(&preview_rgb, &task.preview_format, task.quality)?;
    Ok(DoseBatchTaskResponse {
        task_id: task.task_id.clone(),
        media_type: encoded.media_type,
        content_hex: encoded.content_hex,
        dose_min: round6(dose_min),
        dose_max: round6(dose_max),
        dose_mean: round6(dose_mean),
    })
}

fn compute_dose_gray(
    image: &RgbImage,
    film_background_mean: f64,
    scanner_background_mean: f64,
    film_model: &DoseFilmModel,
    background_quantile: f64,
) -> Result<(GrayImage, f64, f64, f64)> {
    let mut red_values = Vec::with_capacity((image.width() * image.height()) as usize);
    for pixel in image.pixels() {
        red_values.push(f64::from(pixel[0]));
    }
    if red_values.is_empty() {
        return Err(anyhow!("empty patch image"));
    }

    let signal_floor = (film_background_mean - scanner_background_mean).max(1e-6);
    let patch_background_mean = quantile_percentile(&mut red_values.clone(), background_quantile);
    let background_signal = (patch_background_mean - scanner_background_mean).max(1e-6);
    let od_background = (signal_floor / background_signal).max(1.0).log10();
    let baseline = evaluate_curve_model(film_model, od_background)?;

    let mut dose_values = Vec::with_capacity(red_values.len());
    for red in red_values {
        let patch_signal = (red - scanner_background_mean).max(1e-6);
        let od = (signal_floor / patch_signal).max(1.0).log10();
        let dose = (evaluate_curve_model(film_model, od)? - baseline).max(0.0);
        dose_values.push(dose);
    }

    let mut dose_min = f64::INFINITY;
    let mut dose_max = f64::NEG_INFINITY;
    let mut dose_sum = 0.0f64;
    for value in &dose_values {
        dose_min = dose_min.min(*value);
        dose_max = dose_max.max(*value);
        dose_sum += *value;
    }
    if !dose_min.is_finite() {
        dose_min = 0.0;
    }
    if !dose_max.is_finite() {
        dose_max = 0.0;
    }
    let dose_mean = dose_sum / (dose_values.len() as f64);

    let mut gray = GrayImage::new(image.width(), image.height());
    let max_value = dose_max.max(0.0);
    for (index, pixel) in gray.pixels_mut().enumerate() {
        let value = if max_value <= 0.0 {
            0u8
        } else {
            ((dose_values[index] / max_value) * 255.0)
                .clamp(0.0, 255.0)
                .round() as u8
        };
        *pixel = image::Luma([value]);
    }
    Ok((gray, dose_min, dose_max, dose_mean))
}

fn quantile_percentile(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let q = percentile.clamp(0.0, 100.0) / 100.0;
    let target = ((values.len() - 1) as f64 * q).round() as usize;
    values.select_nth_unstable_by(target, |left, right| {
        left.partial_cmp(right).unwrap_or(Ordering::Equal)
    });
    values[target]
}

fn evaluate_curve_model(model: &DoseFilmModel, od: f64) -> Result<f64> {
    match model {
        DoseFilmModel::Polynomial { coefficients } => {
            let mut result = 0.0f64;
            for (power, coefficient) in coefficients.iter().enumerate() {
                result += *coefficient * od.powi(power as i32);
            }
            Ok(result)
        }
        DoseFilmModel::PowerLaw { scale, exponent } => Ok(*scale * od.powf(*exponent)),
    }
}

fn colorize_dose_gray(gray: &GrayImage, palette: &str) -> Result<RgbImage> {
    let normalized = palette.to_lowercase();
    if normalized == "gray" || normalized == "grey" {
        return Ok(gray_to_rgb(gray));
    }
    let mut output = RgbImage::new(gray.width(), gray.height());
    for (x, y, pixel) in gray.enumerate_pixels() {
        let value = f64::from(pixel[0]) / 255.0;
        let mapped = if normalized == "jet" {
            jet_color(value)
        } else if normalized == "turbo" || normalized == "pseudocolor" || normalized == "pseudo"
        {
            turbo_color(value)
        } else {
            return Err(anyhow!("unsupported dose palette: {palette}"));
        };
        output.put_pixel(x, y, Rgb(mapped));
    }
    Ok(output)
}

fn gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let mut output = RgbImage::new(gray.width(), gray.height());
    for (x, y, pixel) in gray.enumerate_pixels() {
        output.put_pixel(x, y, Rgb([pixel[0], pixel[0], pixel[0]]));
    }
    output
}

fn jet_color(value: f64) -> [u8; 3] {
    let x = value.clamp(0.0, 1.0);
    let red = (1.5 - (4.0 * x - 3.0).abs()).clamp(0.0, 1.0);
    let green = (1.5 - (4.0 * x - 2.0).abs()).clamp(0.0, 1.0);
    let blue = (1.5 - (4.0 * x - 1.0).abs()).clamp(0.0, 1.0);
    [
        (red * 255.0).round() as u8,
        (green * 255.0).round() as u8,
        (blue * 255.0).round() as u8,
    ]
}

fn turbo_color(value: f64) -> [u8; 3] {
    let x = value.clamp(0.0, 1.0);
    let r = 0.13572138
        + 4.61539260 * x
        - 42.66032258 * x.powi(2)
        + 132.13108234 * x.powi(3)
        - 152.94239396 * x.powi(4)
        + 59.28637943 * x.powi(5);
    let g = 0.09140261
        + 2.19418839 * x
        + 4.84296658 * x.powi(2)
        - 14.18503333 * x.powi(3)
        + 4.27729857 * x.powi(4)
        + 2.82956604 * x.powi(5);
    let b = 0.10667330
        + 12.64194608 * x
        - 60.58204836 * x.powi(2)
        + 110.36276771 * x.powi(3)
        - 89.90310912 * x.powi(4)
        + 27.34824973 * x.powi(5);
    [
        (r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (b.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn segment_detect(
    image: &RgbImage,
    request: &SegmentDetectRequest,
) -> Result<SegmentDetectResponse> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mask = compute_sheet_mask(image);
    let components = connected_components(&mask, width, height);

    let mut patches = Vec::new();
    let mut component_payloads = Vec::new();
    for (index, component) in components.iter().enumerate() {
        let component_bbox = clip_bbox(
            component.min_x as i32 - request.padding as i32,
            component.min_y as i32 - request.padding as i32,
            component.width as i32 + 2 * request.padding as i32,
            component.height as i32 + 2 * request.padding as i32,
            width,
            height,
        );
        let component_image = crop_image(image, component_bbox);
        let component_geometry =
            estimate_patch_geometry_from_component_mask(&component.mask, component.width, component.height, &component_image);
        let kept = component.area >= request.min_area;
        component_payloads.push(SegmentComponent {
            component_id: (index + 1) as u32,
            area: component.area,
            bbox: component_bbox,
            angle_deg: round4(component_geometry.angle_deg),
            angle_confidence: round4(component_geometry.angle_confidence),
            angle_source: component_geometry.angle_source.to_string(),
            status_flags: component_geometry.status_flags.clone(),
            kept,
        });

        if kept {
            patches.extend(split_component_patches(
                component,
                request.padding as usize,
                request.min_area,
                width,
                height,
                image,
            ));
        }
    }

    if patches.is_empty() {
        return Err(anyhow!("No RCF patches were detected in the input scan"));
    }

    if request.sort_mode == "xy" {
        patches.sort_by_key(|patch| (patch.bbox[0], patch.bbox[1]));
    } else {
        patches.sort_by_key(|patch| (patch.bbox[1], patch.bbox[0]));
    }
    for (index, patch) in patches.iter_mut().enumerate() {
        patch.order = (index + 1) as u32;
    }

    Ok(SegmentDetectResponse {
        component_count: components.len() as u32,
        mask_png_hex: encode_mask_png_hex(&mask, width as u32, height as u32)?,
        patches,
        components: component_payloads,
    })
}

fn split_component_patches(
    component: &Component,
    padding: usize,
    min_area: u32,
    image_width: usize,
    image_height: usize,
    image: &RgbImage,
) -> Vec<SegmentPatch> {
    let longer_side = component.width.max(component.height);
    let shorter_side = component.width.min(component.height).max(1);
    let ratio = longer_side as f32 / shorter_side as f32;
    let split_count = ratio.round().max(1.0) as usize;

    if ratio <= 1.6 || split_count == 1 {
        return component_patch_from_mask(
            &component.mask,
            component.width,
            component.height,
            component.min_x,
            component.min_y,
            padding,
            image_width,
            image_height,
            image,
        )
        .into_iter()
        .collect();
    }

    let mut patches = Vec::new();
    if component.height >= component.width {
        for index in 0..split_count {
            let local_y0 =
                ((index as f32) * component.height as f32 / split_count as f32).round() as usize;
            let local_y1 = if index == split_count - 1 {
                component.height
            } else {
                (((index + 1) as f32) * component.height as f32 / split_count as f32).round()
                    as usize
            };
            let segment_height = local_y1.saturating_sub(local_y0);
            if segment_height == 0 {
                continue;
            }
            let mut segment_mask = vec![0u8; component.width * segment_height];
            let mut area = 0u32;
            for y in 0..segment_height {
                let src_offset = (local_y0 + y) * component.width;
                let dst_offset = y * component.width;
                let src = &component.mask[src_offset..src_offset + component.width];
                let dst = &mut segment_mask[dst_offset..dst_offset + component.width];
                dst.copy_from_slice(src);
                area += src.iter().map(|value| *value as u32).sum::<u32>();
            }
            if area < (min_area / 4).max(1) {
                continue;
            }
            if let Some(patch) = component_patch_from_mask(
                &segment_mask,
                component.width,
                segment_height,
                component.min_x,
                component.min_y + local_y0,
                padding,
                image_width,
                image_height,
                image,
            ) {
                patches.push(patch);
            }
        }
    } else {
        for index in 0..split_count {
            let local_x0 =
                ((index as f32) * component.width as f32 / split_count as f32).round() as usize;
            let local_x1 = if index == split_count - 1 {
                component.width
            } else {
                (((index + 1) as f32) * component.width as f32 / split_count as f32).round()
                    as usize
            };
            let segment_width = local_x1.saturating_sub(local_x0);
            if segment_width == 0 {
                continue;
            }
            let mut segment_mask = vec![0u8; segment_width * component.height];
            let mut area = 0u32;
            for y in 0..component.height {
                for x in 0..segment_width {
                    let value = component.mask[y * component.width + local_x0 + x];
                    segment_mask[y * segment_width + x] = value;
                    area += value as u32;
                }
            }
            if area < (min_area / 4).max(1) {
                continue;
            }
            if let Some(patch) = component_patch_from_mask(
                &segment_mask,
                segment_width,
                component.height,
                component.min_x + local_x0,
                component.min_y,
                padding,
                image_width,
                image_height,
                image,
            ) {
                patches.push(patch);
            }
        }
    }

    if patches.is_empty() {
        component_patch_from_mask(
            &component.mask,
            component.width,
            component.height,
            component.min_x,
            component.min_y,
            padding,
            image_width,
            image_height,
            image,
        )
        .into_iter()
        .collect()
    } else {
        patches
    }
}

fn component_patch_from_mask(
    mask: &[u8],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    padding: usize,
    image_width: usize,
    image_height: usize,
    image: &RgbImage,
) -> Option<SegmentPatch> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] == 0 {
                continue;
            }
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !found {
        return None;
    }

    let bbox = clip_bbox(
        origin_x as i32 + min_x as i32 - padding as i32,
        origin_y as i32 + min_y as i32 - padding as i32,
        (max_x - min_x + 1) as i32 + 2 * padding as i32,
        (max_y - min_y + 1) as i32 + 2 * padding as i32,
        image_width,
        image_height,
    );
    let patch_image = crop_image(image, bbox);
    let geometry = estimate_patch_geometry_from_component_mask(mask, width, height, &patch_image);
    Some(SegmentPatch {
        order: 0,
        bbox,
        angle_deg: round4(geometry.angle_deg),
        angle_confidence: round4(geometry.angle_confidence),
        angle_source: geometry.angle_source.to_string(),
        status_flags: geometry.status_flags,
    })
}

fn estimate_patch_geometry(patch_image: &RgbImage) -> GeometryEstimate {
    let background = estimate_patch_background(patch_image);
    let mask = compute_patch_film_mask(patch_image, background);
    pca_geometry_from_mask(
        &mask,
        patch_image.width() as usize,
        patch_image.height() as usize,
    )
}

fn estimate_patch_geometry_from_component_mask(
    component_mask: &[u8],
    component_width: usize,
    component_height: usize,
    patch_image: &RgbImage,
) -> GeometryEstimate {
    let component_area = component_mask.iter().map(|value| *value as u32).sum::<u32>() as usize;
    let contour_geometry = estimate_patch_geometry(patch_image);
    let background = estimate_patch_background(patch_image);
    let film_mask = compute_patch_film_mask(patch_image, background);
    let film_area = film_mask.iter().map(|value| *value as u32).sum::<u32>() as usize;
    let min_film_area = 64usize.max(component_area / 5);
    if film_area >= min_film_area && contour_geometry.angle_source != "low_confidence_zero" {
        return contour_geometry;
    }
    pca_geometry_from_mask(component_mask, component_width, component_height)
}

fn pca_geometry_from_mask(mask: &[u8], width: usize, height: usize) -> GeometryEstimate {
    let mut count = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;

    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] == 0 {
                continue;
            }
            count += 1.0;
            sum_x += x as f64;
            sum_y += y as f64;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    if count < 2.0 {
        return GeometryEstimate {
            angle_deg: 0.0,
            angle_confidence: 0.0,
            angle_source: "low_confidence_zero",
            status_flags: vec!["low_confidence_angle".to_string()],
        };
    }

    let mean_x = sum_x / count;
    let mean_y = sum_y / count;
    let mut cov_xx = 0.0f64;
    let mut cov_xy = 0.0f64;
    let mut cov_yy = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] == 0 {
                continue;
            }
            let dx = x as f64 - mean_x;
            let dy = y as f64 - mean_y;
            cov_xx += dx * dx;
            cov_xy += dx * dy;
            cov_yy += dy * dy;
        }
    }
    cov_xx /= count;
    cov_xy /= count;
    cov_yy /= count;

    let angle_deg =
        fold_rect_angle((0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy)).to_degrees() as f32);
    let trace = cov_xx + cov_yy;
    let delta = ((cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy).sqrt();
    let lambda1 = (trace + delta) / 2.0;
    let lambda2 = (trace - delta) / 2.0;
    let anisotropy = if lambda1 > 1e-6 {
        ((lambda1 - lambda2) / lambda1).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let bbox_width = max_x - min_x + 1;
    let bbox_height = max_y - min_y + 1;
    let aspect_ratio =
        bbox_width.max(bbox_height) as f64 / bbox_width.min(bbox_height).max(1) as f64;

    if aspect_ratio < 1.12 && angle_deg.abs() < 5.0 {
        return GeometryEstimate {
            angle_deg: 0.0,
            angle_confidence: 0.0,
            angle_source: "low_confidence_zero",
            status_flags: vec!["low_confidence_angle".to_string()],
        };
    }

    let confidence = (0.7 * anisotropy + 0.3 * (((aspect_ratio - 1.0) / 0.35).clamp(0.0, 1.0)))
        .clamp(0.05, 1.0) as f32;
    GeometryEstimate {
        angle_deg,
        angle_confidence: confidence,
        angle_source: "contour_rect",
        status_flags: Vec::new(),
    }
}

fn compute_sheet_mask(image: &RgbImage) -> Vec<u8> {
    let background = border_median_rgb(
        image,
        border_width(image.width() as usize, image.height() as usize),
    );
    let border_pixels = border_rgb_pixels(
        image,
        border_width(image.width() as usize, image.height() as usize),
    );
    let mut border_distances = border_pixels
        .iter()
        .map(|pixel| color_distance(*pixel, background))
        .collect::<Vec<_>>();
    let threshold = percentile(&mut border_distances, 99.0)
        .mul_add(3.0, 0.0)
        .max(8.0);
    let mut mask = threshold_mask(image, background, threshold);
    let width = image.width() as usize;
    let height = image.height() as usize;
    mask = close_square(&mask, width, height, 3);
    mask = open_square(&mask, width, height, 1);
    fill_holes(&mask, width, height)
}

fn compute_patch_film_mask(image: &RgbImage, background: [f32; 3]) -> Vec<u8> {
    let border = border_width(image.width() as usize, image.height() as usize);
    let border_pixels = border_rgb_pixels(image, border);
    let mut border_distances = border_pixels
        .iter()
        .map(|pixel| color_distance(*pixel, background))
        .collect::<Vec<_>>();
    let threshold = percentile(&mut border_distances, 98.0)
        .mul_add(2.5, 0.0)
        .max(4.0);
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut mask = threshold_mask(image, background, threshold);
    mask = close_square(&mask, width, height, 2);
    mask = open_square(&mask, width, height, 1);
    mask = fill_holes(&mask, width, height);
    largest_component_mask(&mask, width, height)
}

fn threshold_mask(image: &RgbImage, background: [f32; 3], threshold: f32) -> Vec<u8> {
    image
        .pixels()
        .map(|pixel| {
            if color_distance([pixel[0], pixel[1], pixel[2]], background) >= threshold {
                1
            } else {
                0
            }
        })
        .collect()
}

fn estimate_patch_background(image: &RgbImage) -> [f32; 3] {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let border = border_width(width, height)
        .min((height / 4).max(1))
        .min((width / 4).max(1));
    border_median_rgb(image, border.max(1))
}

fn border_width(width: usize, height: usize) -> usize {
    (width.min(height) / 20).max(4)
}

fn border_rgb_pixels(image: &RgbImage, border: usize) -> Vec<[u8; 3]> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let border = border.min(width.max(1)).min(height.max(1)).max(1);
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if x < border || y < border || x >= width - border || y >= height - border {
                let pixel = image.get_pixel(x as u32, y as u32);
                pixels.push([pixel[0], pixel[1], pixel[2]]);
            }
        }
    }
    pixels
}

fn border_median_rgb(image: &RgbImage, border: usize) -> [f32; 3] {
    let pixels = border_rgb_pixels(image, border);
    let mut red = pixels.iter().map(|pixel| pixel[0]).collect::<Vec<_>>();
    let mut green = pixels.iter().map(|pixel| pixel[1]).collect::<Vec<_>>();
    let mut blue = pixels.iter().map(|pixel| pixel[2]).collect::<Vec<_>>();
    [
        median_u8(&mut red) as f32,
        median_u8(&mut green) as f32,
        median_u8(&mut blue) as f32,
    ]
}

fn median_u8(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn percentile(values: &mut [f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((percentile / 100.0) * (values.len().saturating_sub(1)) as f32).round() as usize;
    values[rank.min(values.len() - 1)]
}

fn color_distance(pixel: [u8; 3], background: [f32; 3]) -> f32 {
    let dr = pixel[0] as f32 - background[0];
    let dg = pixel[1] as f32 - background[1];
    let db = pixel[2] as f32 - background[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

fn close_square(mask: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    erode_square(
        &dilate_square(mask, width, height, radius),
        width,
        height,
        radius,
    )
}

fn open_square(mask: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    dilate_square(
        &erode_square(mask, width, height, radius),
        width,
        height,
        radius,
    )
}

fn dilate_square(mask: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    if radius == 0 {
        return mask.to_vec();
    }
    let integral = integral_mask(mask, width, height);
    let stride = width + 1;
    let mut output = vec![0u8; mask.len()];
    for y in 0..height {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(height);
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(width);
            if rect_sum(&integral, stride, x0, y0, x1, y1) > 0 {
                output[y * width + x] = 1;
            }
        }
    }
    output
}

fn erode_square(mask: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    if radius == 0 {
        return mask.to_vec();
    }
    let integral = integral_mask(mask, width, height);
    let stride = width + 1;
    let full_area = ((2 * radius + 1) * (2 * radius + 1)) as u32;
    let mut output = vec![0u8; mask.len()];
    for y in 0..height {
        for x in 0..width {
            if x < radius || y < radius || x + radius >= width || y + radius >= height {
                continue;
            }
            let x0 = x - radius;
            let y0 = y - radius;
            let x1 = x + radius + 1;
            let y1 = y + radius + 1;
            if rect_sum(&integral, stride, x0, y0, x1, y1) == full_area {
                output[y * width + x] = 1;
            }
        }
    }
    output
}

fn integral_mask(mask: &[u8], width: usize, height: usize) -> Vec<u32> {
    let stride = width + 1;
    let mut integral = vec![0u32; stride * (height + 1)];
    for y in 0..height {
        let mut row_sum = 0u32;
        for x in 0..width {
            row_sum += mask[y * width + x] as u32;
            integral[(y + 1) * stride + (x + 1)] = integral[y * stride + (x + 1)] + row_sum;
        }
    }
    integral
}

fn rect_sum(integral: &[u32], stride: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> u32 {
    integral[y1 * stride + x1] + integral[y0 * stride + x0]
        - integral[y0 * stride + x1]
        - integral[y1 * stride + x0]
}

fn fill_holes(mask: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();

    for x in 0..width {
        enqueue_zero(mask, width, height, x, 0, &mut visited, &mut queue);
        enqueue_zero(
            mask,
            width,
            height,
            x,
            height.saturating_sub(1),
            &mut visited,
            &mut queue,
        );
    }
    for y in 0..height {
        enqueue_zero(mask, width, height, 0, y, &mut visited, &mut queue);
        enqueue_zero(
            mask,
            width,
            height,
            width.saturating_sub(1),
            y,
            &mut visited,
            &mut queue,
        );
    }

    while let Some((x, y)) = queue.pop_front() {
        if x > 0 {
            enqueue_zero(mask, width, height, x - 1, y, &mut visited, &mut queue);
        }
        if y > 0 {
            enqueue_zero(mask, width, height, x, y - 1, &mut visited, &mut queue);
        }
        if x + 1 < width {
            enqueue_zero(mask, width, height, x + 1, y, &mut visited, &mut queue);
        }
        if y + 1 < height {
            enqueue_zero(mask, width, height, x, y + 1, &mut visited, &mut queue);
        }
    }

    let mut output = mask.to_vec();
    for (index, value) in output.iter_mut().enumerate() {
        if *value == 0 && !visited[index] {
            *value = 1;
        }
    }
    output
}

fn enqueue_zero(
    mask: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    visited: &mut [bool],
    queue: &mut VecDeque<(usize, usize)>,
) {
    if x >= width || y >= height {
        return;
    }
    let index = y * width + x;
    if mask[index] != 0 || visited[index] {
        return;
    }
    visited[index] = true;
    queue.push_back((x, y));
}

fn largest_component_mask(mask: &[u8], width: usize, height: usize) -> Vec<u8> {
    let components = connected_components(mask, width, height);
    if components.is_empty() {
        return vec![0u8; mask.len()];
    }
    let largest = components
        .iter()
        .max_by_key(|component| component.area)
        .expect("components not empty");
    let mut output = vec![0u8; mask.len()];
    for y in 0..largest.height {
        for x in 0..largest.width {
            if largest.mask[y * largest.width + x] == 0 {
                continue;
            }
            output[(largest.min_y + y) * width + largest.min_x + x] = 1;
        }
    }
    output
}

fn connected_components(mask: &[u8], width: usize, height: usize) -> Vec<Component> {
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    let mut components = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if mask[index] == 0 || visited[index] {
                continue;
            }

            visited[index] = true;
            queue.push_back((x, y));
            let mut pixels = Vec::new();
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;

            while let Some((cx, cy)) = queue.pop_front() {
                pixels.push((cx, cy));
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);

                let neighbors = [
                    (cx.wrapping_sub(1), cy, cx > 0),
                    (cx + 1, cy, cx + 1 < width),
                    (cx, cy.wrapping_sub(1), cy > 0),
                    (cx, cy + 1, cy + 1 < height),
                ];
                for (nx, ny, valid) in neighbors {
                    if !valid {
                        continue;
                    }
                    let neighbor_index = ny * width + nx;
                    if mask[neighbor_index] == 0 || visited[neighbor_index] {
                        continue;
                    }
                    visited[neighbor_index] = true;
                    queue.push_back((nx, ny));
                }
            }

            let component_width = max_x - min_x + 1;
            let component_height = max_y - min_y + 1;
            let mut component_mask = vec![0u8; component_width * component_height];
            for (px, py) in pixels.iter().copied() {
                component_mask[(py - min_y) * component_width + (px - min_x)] = 1;
            }
            components.push(Component {
                area: pixels.len() as u32,
                min_x,
                min_y,
                mask: component_mask,
                width: component_width,
                height: component_height,
            });
        }
    }

    components
}

fn clip_bbox(
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    image_width: usize,
    image_height: usize,
) -> [u32; 4] {
    let x = x.max(0).min(image_width.saturating_sub(1) as i32);
    let y = y.max(0).min(image_height.saturating_sub(1) as i32);
    let width = width.max(1).min(image_width as i32 - x);
    let height = height.max(1).min(image_height as i32 - y);
    [x as u32, y as u32, width as u32, height as u32]
}

fn encode_mask_png_hex(mask: &[u8], width: u32, height: u32) -> Result<String> {
    let mut gray = GrayImage::new(width, height);
    for (index, pixel) in gray.pixels_mut().enumerate() {
        let value = if mask[index] == 0 { 0u8 } else { 255u8 };
        *pixel = image::Luma([value]);
    }
    let mut bytes = Vec::new();
    let encoder = PngEncoder::new(&mut bytes);
    encoder.write_image(gray.as_raw(), width, height, ColorType::L8.into())?;
    Ok(hex::encode(bytes))
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn order_points(points: &[[f32; 2]]) -> Result<[[f32; 2]; 4]> {
    if points.len() != 4 {
        return Err(anyhow!("expected four points"));
    }
    let mut top_left = points[0];
    let mut bottom_right = points[0];
    let mut top_right = points[0];
    let mut bottom_left = points[0];
    let mut min_sum = f32::INFINITY;
    let mut max_sum = f32::NEG_INFINITY;
    let mut min_diff = f32::INFINITY;
    let mut max_diff = f32::NEG_INFINITY;

    for point in points {
        let sum = point[0] + point[1];
        let diff = point[1] - point[0];
        if sum < min_sum {
            min_sum = sum;
            top_left = *point;
        }
        if sum > max_sum {
            max_sum = sum;
            bottom_right = *point;
        }
        if diff < min_diff {
            min_diff = diff;
            top_right = *point;
        }
        if diff > max_diff {
            max_diff = diff;
            bottom_left = *point;
        }
    }

    Ok([top_left, top_right, bottom_right, bottom_left])
}

fn fold_rect_angle(angle_deg: f32) -> f32 {
    let mut normalized = ((angle_deg + 90.0).rem_euclid(180.0)) - 90.0;
    if normalized == -90.0 {
        normalized = 90.0;
    }
    if normalized > 45.0 {
        normalized - 90.0
    } else if normalized < -45.0 {
        normalized + 90.0
    } else {
        normalized
    }
}

fn round4(value: f32) -> f32 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::{
        clip_bbox, close_square, component_patch_from_mask, compute_patch_film_mask,
        compute_sheet_mask, crop_image, order_points, pca_geometry_from_mask,
        resize_for_preview, segment_detect, SegmentDetectRequest,
    };
    use image::{Rgb, RgbImage};
    use imageproc::drawing::draw_polygon_mut;
    use imageproc::point::Point;

    #[test]
    fn order_points_normalizes_arbitrary_quad_order() {
        let ordered = order_points(&[[200.0, 20.0], [20.0, 200.0], [20.0, 20.0], [200.0, 200.0]])
            .expect("quad should order correctly");
        assert_eq!(ordered[0], [20.0, 20.0]);
        assert_eq!(ordered[1], [200.0, 20.0]);
        assert_eq!(ordered[2], [200.0, 200.0]);
        assert_eq!(ordered[3], [20.0, 200.0]);
    }

    #[test]
    fn resize_for_preview_caps_largest_dimension() {
        let image = RgbImage::from_pixel(3200, 1800, Rgb([10, 20, 30]));
        let resized = resize_for_preview(&image, 320);
        assert_eq!(resized.dimensions(), (320, 180));
    }

    #[test]
    fn crop_image_clamps_bbox_to_image_bounds() {
        let image = RgbImage::from_pixel(100, 80, Rgb([10, 20, 30]));
        let cropped = crop_image(&image, [80, 60, 50, 30]);
        assert_eq!(cropped.dimensions(), (20, 20));
    }

    #[test]
    fn close_square_bridges_small_gap() {
        let mut mask = vec![0u8; 5 * 5];
        for y in 1..4 {
            mask[y * 5 + 0] = 1;
            mask[y * 5 + 1] = 1;
            mask[y * 5 + 3] = 1;
            mask[y * 5 + 4] = 1;
        }
        let closed = close_square(&mask, 5, 5, 1);
        assert_eq!(closed[2 * 5 + 2], 1);
    }

    #[test]
    fn pca_geometry_returns_zero_for_axis_aligned_rectangle() {
        let mut mask = vec![0u8; 20 * 12];
        for y in 2..10 {
            for x in 3..17 {
                mask[y * 20 + x] = 1;
            }
        }
        let geometry = pca_geometry_from_mask(&mask, 20, 12);
        assert_eq!(geometry.angle_source, "contour_rect");
        assert!(geometry.angle_deg.abs() < 1.0);
        assert!(geometry.angle_confidence > 0.0);
    }

    #[test]
    fn compute_sheet_mask_detects_low_contrast_rectangle() {
        let mut image = RgbImage::from_pixel(80, 60, Rgb([245, 245, 245]));
        for y in 10..50 {
            for x in 15..55 {
                image.put_pixel(x, y, Rgb([238, 240, 240]));
            }
        }
        let mask = compute_sheet_mask(&image);
        let detected = mask.iter().map(|value| *value as u32).sum::<u32>();
        assert!(detected > 1200);
    }

    #[test]
    fn compute_patch_film_mask_keeps_sheet_not_dark_center_only() {
        let mut image = RgbImage::from_pixel(80, 80, Rgb([245, 245, 245]));
        for y in 12..68 {
            for x in 12..68 {
                image.put_pixel(x, y, Rgb([238, 240, 240]));
            }
        }
        for y in 30..50 {
            for x in 30..50 {
                image.put_pixel(x, y, Rgb([120, 150, 180]));
            }
        }
        let mask = compute_patch_film_mask(&image, [245.0, 245.0, 245.0]);
        let detected = mask.iter().map(|value| *value as u32).sum::<u32>();
        assert!(detected > 2500);
    }

    #[test]
    fn clip_bbox_clamps_to_image() {
        let bbox = clip_bbox(-5, 10, 40, 40, 30, 25);
        assert_eq!(bbox, [0, 10, 30, 15]);
    }

    #[test]
    fn component_patch_from_mask_preserves_rotated_component_angle() {
        let width = 420usize;
        let height = 300usize;
        let mut image = RgbImage::from_pixel(width as u32, height as u32, Rgb([245, 245, 245]));
        let mut mask = vec![0u8; width * height];
        let polygon = vec![
            Point::new(180, 50),
            Point::new(300, 90),
            Point::new(250, 240),
            Point::new(130, 200),
        ];
        draw_polygon_mut(&mut image, &polygon, Rgb([215, 218, 200]));
        draw_polygon_mut(
            &mut image,
            &[
                Point::new(200, 110),
                Point::new(255, 125),
                Point::new(238, 182),
                Point::new(182, 168),
            ],
            Rgb([120, 150, 180]),
        );
        for point in &polygon {
            let x = point.x.clamp(0, (width - 1) as i32) as usize;
            let y = point.y.clamp(0, (height - 1) as i32) as usize;
            mask[y * width + x] = 1;
        }
        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x as u32, y as u32);
                if pixel[0] != 245 || pixel[1] != 245 || pixel[2] != 245 {
                    mask[y * width + x] = 1;
                }
            }
        }

        let patch = component_patch_from_mask(
            &mask,
            width,
            height,
            0,
            0,
            4,
            width,
            height,
            &image,
        )
        .expect("rotated component should produce a patch");

        assert_eq!(patch.angle_source, "contour_rect");
        assert!(patch.angle_deg.abs() >= 5.0);
        assert!(patch.angle_confidence > 0.0);
    }

    #[test]
    fn segment_detect_keeps_rotated_patch_angle() {
        let mut image = RgbImage::from_pixel(420, 300, Rgb([245, 245, 245]));
        draw_polygon_mut(
            &mut image,
            &[
                Point::new(180, 50),
                Point::new(300, 90),
                Point::new(250, 240),
                Point::new(130, 200),
            ],
            Rgb([215, 218, 200]),
        );
        let request = SegmentDetectRequest {
            scan_file: "synthetic".to_string(),
            min_area: 1000,
            padding: 4,
            sort_mode: "yx".to_string(),
        };

        let response = segment_detect(&image, &request).expect("segment detect should succeed");

        assert_eq!(response.patches.len(), 1);
        assert_eq!(response.patches[0].angle_source, "contour_rect");
        assert!(response.patches[0].angle_deg.abs() >= 5.0);
        assert!(response.patches[0].angle_confidence > 0.0);
    }
}
