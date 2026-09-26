"use strict";

const $ = (id) => document.getElementById(id);

function num(value, digits = 2) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(digits);
}

function formatMatrix(matrix) {
  return matrix
    .map((row) => "  " + row.map((v) => Number(v).toFixed(3)).join("  "))
    .join("\n");
}

function renderCalibration(cal, proj) {
  $("c-detected").textContent = `${cal.num_detected}/${cal.num_images}`;
  $("c-rms").textContent = num(cal.rms, 4) + " px";
  $("c-mean").textContent = num(cal.mean_reprojection, 4) + " px";
  $("c-fx").textContent = num(proj.fx, 1) + " px";

  $("c-matrix").textContent = formatMatrix(cal.camera_matrix);
  $("c-dist").textContent = "  " + cal.distortion.map((v) => Number(v).toFixed(6)).join("  ");
  $("c-scale").textContent =
    `fx / Z = ${num(proj.pixels_per_cm, 3)} px/cm  ·  object distance Z = ${num(proj.object_distance_m, 2)} m`;

  const strip = $("corner-strip");
  strip.innerHTML = "";
  cal.corner_thumbs.forEach((dataUrl, i) => {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = dataUrl;
    img.alt = cal.per_image[i] ? cal.per_image[i].name : `image ${i}`;
    const cap = document.createElement("figcaption");
    cap.textContent = cal.per_image[i] ? cal.per_image[i].name : `image ${i}`;
    fig.appendChild(img);
    fig.appendChild(cap);
    strip.appendChild(fig);
  });
}

function renderMeasurements(measurements) {
  const container = $("measurement-cards");
  container.innerHTML = "";

  measurements.forEach((m) => {
    const section = document.createElement("section");
    section.className = "measurement";

    const heading = document.createElement("h3");
    heading.innerHTML = `<span class="tag">${m.image}</span> · ${m.detected_bars} bars detected`;

    const panels = document.createElement("div");
    panels.className = "panels";
    const panelDefs = [
      ["Undistorted image", m.undistorted],
      ["Detected paper boundary", m.paper_corners],
      ["Rectified bars", m.result],
      ["Binary mask", m.binary],
      ["Horizontal structure", m.horizontal],
    ];
    panelDefs.forEach(([label, src]) => {
      const fig = document.createElement("figure");
      fig.className = "panel";
      const img = document.createElement("img");
      img.src = src;
      img.alt = label;
      const cap = document.createElement("figcaption");
      cap.textContent = label;
      fig.appendChild(img);
      fig.appendChild(cap);
      panels.appendChild(fig);
    });

    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    let html =
      "<thead><tr><th>#</th><th>Measured (cm)</th><th>Truth (cm)</th><th>Error (cm)</th><th>% error</th></tr></thead><tbody>";
    m.bars.forEach((row) => {
      const errCls = row.error_cm >= 0 ? "err-pos" : "err-neg";
      html += `<tr><td>${row.index}</td><td>${row.measured_cm.toFixed(3)}</td><td>${row.truth_cm}</td>` +
        `<td class="${errCls}">${row.error_cm >= 0 ? "+" : ""}${row.error_cm.toFixed(3)}</td>` +
        `<td class="${errCls}">${row.pct_error >= 0 ? "+" : ""}${row.pct_error.toFixed(2)}%</td></tr>`;
    });
    html += "</tbody></table>";
    table.innerHTML = html;
    wrap.appendChild(table);

    const paperCheck = document.createElement("p");
    paperCheck.className = "paper-check";
    paperCheck.textContent =
      `Paper sanity check: measured ${m.paper_w_cm} × ${m.paper_h_cm} cm  ·  ` +
      `truth ${m.paper_truth_cm[0]} × ${m.paper_truth_cm[1]} cm`;

    section.appendChild(heading);
    section.appendChild(panels);
    section.appendChild(wrap);
    section.appendChild(paperCheck);
    container.appendChild(section);
  });
}

function renderValidation(stats, errorPlot) {
  $("v-mae").textContent = num(stats.mean_abs, 3) + " cm";
  $("v-rmse").textContent = num(stats.rmse, 3) + " cm";
  $("v-std").textContent = num(stats.std, 3) + " cm";
  $("v-max").textContent = num(stats.max_abs, 3) + " cm";
  $("error-plot").src = errorPlot;
}

async function loadResults() {
  try {
    const resp = await fetch("/api/results");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    renderCalibration(data.calibration, data.projection);
    renderMeasurements(data.measurements);
    renderValidation(data.stats, data.error_plot);

    $("calibration").hidden = false;
    $("measurements").hidden = false;
    $("validation").hidden = false;
  } catch (err) {
    const el = $("error");
    el.textContent = "Error: " + err.message;
    el.hidden = false;
  }
}

loadResults();
