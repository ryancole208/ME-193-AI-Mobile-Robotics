// Debug page client logic for the Sumo Opponent Tracker.
//
// Sliders are generated from small spec arrays below rather than written
// out by hand in HTML, so adding a new tunable later is a one-line change
// here instead of editing both index.html and this file.

const SIMPLE_SLIDERS = [
  { path: ['blur_ksize'], label: 'Blur kernel size', min: 1, max: 15, step: 2 },
  { path: ['morph_open_ksize'], label: 'Morph open size (noise removal)', min: 1, max: 15, step: 2 },
  { path: ['morph_close_ksize'], label: 'Morph close size (merge robot parts)', min: 1, max: 41, step: 2 },
  { path: ['min_area_frac'], label: 'Min blob area (fraction of frame)', min: 0, max: 0.2, step: 0.001 },
  { path: ['max_area_frac'], label: 'Max blob area (fraction of frame)', min: 0, max: 1, step: 0.01 },
  { path: ['min_aspect_ratio'], label: 'Min aspect ratio (rejects thin tape)', min: 0, max: 1, step: 0.01 },
  { path: ['min_fill_ratio'], label: 'Min fill ratio (rejects thin/diagonal tape)', min: 0, max: 1, step: 0.01 },
  { path: ['tape_color_margin'], label: 'Tape color match margin', min: 0, max: 60, step: 1 },
  { path: ['debounce_frames'], label: 'Debounce frames', min: 1, max: 10, step: 1 },
  { path: ['lost_grace_s'], label: 'Lost-target grace period (s)', min: 0, max: 2, step: 0.05 },
  { path: ['min_saturation_colored'], label: 'Min saturation for azure/pink (filters gray/white false positives)', min: 0, max: 255, step: 1 },
  { path: ['min_value_for_hue'], label: 'Min brightness for azure/pink (filters near-black noise)', min: 0, max: 255, step: 1 },
  { path: ['mask_saturated_warning_pct'], label: 'Exclude a color mask if it covers more than this % of the ROI', min: 5, max: 100, step: 1 },
];

const COLOR_NAMES = ['azure', 'pink', 'white', 'wheel'];
const CHANNELS = [
  { key: 'h', label: 'Hue', max: 179 },
  { key: 's', label: 'Sat', max: 255 },
  { key: 'v', label: 'Val', max: 255 },
];

// Every "Set <name> here" button, in the order they're shown.
const LEARN_TARGETS = ['azure', 'pink', 'white', 'wheel', 'floor', 'tape'];

// Options in the "what the detector sees" mask dropdown.
const MASK_CHANNELS = ['combined', 'azure', 'pink', 'white', 'wheel', 'non_floor'];

let config = null; // the full config dict, kept in sync with the server

const ui = new WebUI();

const statusEl = document.getElementById('status');
const liveImage = document.getElementById('liveImage');
const maskImage = document.getElementById('maskImage');

ui.on_connect(() => {
  statusEl.textContent = 'connected, waiting for config and frames...';
});
ui.on_disconnect(() => {
  statusEl.textContent = 'disconnected from the board';
});

// Click the live image to move the sample box there - the orange box on
// the next frame will jump to wherever you clicked.
liveImage.addEventListener('click', event => {
  const rect = liveImage.getBoundingClientRect();
  const x_frac = (event.clientX - rect.left) / rect.width;
  const y_frac = (event.clientY - rect.top) / rect.height;
  ui.send_message('set_sample_point', { x_frac, y_frac });
});

renderLearnButtons();
renderMaskChannelSelect();

function renderLearnButtons() {
  const container = document.getElementById('learnButtons');
  container.className = 'learn-buttons';
  LEARN_TARGETS.forEach(target => {
    const button = document.createElement('button');
    button.textContent = `Set ${target} here`;
    button.addEventListener('click', () => ui.send_message('learn_color', { target }));
    container.appendChild(button);
  });
}

function renderMaskChannelSelect() {
  const select = document.getElementById('maskChannelSelect');
  MASK_CHANNELS.forEach(channel => {
    const option = document.createElement('option');
    option.value = channel;
    option.textContent = channel;
    select.appendChild(option);
  });
  select.addEventListener('change', () => {
    ui.send_message('set_mask_channel', { channel: select.value });
  });
}

document.getElementById('floorFallbackCheckbox').addEventListener('change', event => {
  sendConfigUpdate(['floor_fallback_enabled'], event.target.checked);
});

document.getElementById('resetColorsButton').addEventListener('click', () => ui.send_message('reset_colors'));
document.getElementById('setHorizonButton').addEventListener('click', () => ui.send_message('set_line', { line: 'horizon' }));
document.getElementById('setOwnRobotButton').addEventListener('click', () => ui.send_message('set_line', { line: 'own_robot' }));

ui.on_message('config', data => {
  config = data;
  renderSliders();
  renderLearnedColors();
  renderDistancePoints();
  document.getElementById('floorFallbackCheckbox').checked = !!config.floor_fallback_enabled;
});

ui.on_message('frame', data => {
  if (data.image) liveImage.src = 'data:image/jpeg;base64,' + data.image;
  if (data.mask) maskImage.src = 'data:image/jpeg;base64,' + data.mask;

  const d = data.detection;
  // x and distance_cm are null while the opponent isn't visible
  const xStr = d.x != null ? d.x.toFixed(2) : 'null';
  const distanceStr = d.distance_cm != null ? `${d.distance_cm.toFixed(0)}cm` : 'null';
  let text =
    `source=${data.source}  state=${data.state}  fps=${data.fps.toFixed(1)}  frame_age=${data.frame_age.toFixed(2)}s\n` +
    `visible=${d.visible}  x=${xStr}  distance_cm=${distanceStr}  confidence=${d.confidence.toFixed(2)}  ` +
    `groups=${d.group_count}  group_pixels=${d.group_pixels}  proc=${d.processing_ms.toFixed(1)}ms`;
  if (data.warnings && data.warnings.length) {
    text += '\n' + data.warnings.map(w => `⚠ ${w}`).join('\n');
  }
  statusEl.textContent = text;
});

ui.on_message('calibration_result', data => {
  const el = document.getElementById(data.kind === 'distance' ? 'distanceStatus' : 'learnedColors');
  const prefix = data.success ? '✓' : '✗';
  el.textContent = `${prefix} ${data.message}`;
});

ui.on_message('record_result', data => {
  const el = document.getElementById('recordStatus');
  el.textContent = data.success ? `Saved ${data.filename}` : `Could not record: ${data.error}`;
});

ui.on_message('recordings_list', data => {
  const list = document.getElementById('recordingsList');
  list.innerHTML = '';
  data.files.forEach(filename => {
    const li = document.createElement('li');
    const replayButton = document.createElement('button');
    replayButton.textContent = 'Replay';
    replayButton.addEventListener('click', () => ui.send_message('replay_frame', { filename }));
    const label = document.createElement('span');
    label.textContent = filename;
    li.appendChild(replayButton);
    li.appendChild(label);
    list.appendChild(li);
  });
});

function getPath(obj, path) {
  return path.reduce((node, key) => (node == null ? undefined : node[key]), obj);
}

function setPathLocally(path, value) {
  let node = config;
  for (const key of path.slice(0, -1)) node = node[key];
  node[path[path.length - 1]] = value;
}

function sendConfigUpdate(path, value) {
  setPathLocally(path, value);
  ui.send_message('update_config', { path, value });
}

function renderSliders() {
  const container = document.getElementById('simpleSliders');
  container.innerHTML = '';
  SIMPLE_SLIDERS.forEach(spec => {
    const value = getPath(config, spec.path);
    const row = document.createElement('div');
    row.className = 'slider-row';

    const label = document.createElement('label');
    label.textContent = spec.label;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step;
    input.value = value;

    const out = document.createElement('output');
    out.textContent = value;

    input.addEventListener('input', () => {
      const parsed = Number(input.value);
      out.textContent = parsed;
      sendConfigUpdate(spec.path, parsed);
    });

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(out);
    container.appendChild(row);
  });

  renderColorSliders();
}

function renderColorSliders() {
  const container = document.getElementById('colorSliders');
  container.innerHTML = '';

  COLOR_NAMES.forEach(colorName => {
    const colorConfig = config.colors[colorName];
    const block = document.createElement('div');
    block.className = 'color-block';

    const title = document.createElement('h4');
    title.textContent = colorName;
    block.appendChild(title);

    CHANNELS.forEach(channel => {
      const [lo, hi] = colorConfig[channel.key];
      const row = document.createElement('div');
      row.className = 'channel-row';

      const label = document.createElement('label');
      label.textContent = channel.label;

      const pair = document.createElement('div');
      pair.className = 'pair';

      const loInput = makeRangeInput(0, channel.max, lo);
      const loOut = document.createElement('output');
      loOut.textContent = lo;
      const hiInput = makeRangeInput(0, channel.max, hi);
      const hiOut = document.createElement('output');
      hiOut.textContent = hi;

      const update = () => {
        const newLo = Number(loInput.value);
        const newHi = Number(hiInput.value);
        loOut.textContent = newLo;
        hiOut.textContent = newHi;
        sendConfigUpdate(['colors', colorName, channel.key], [newLo, newHi]);
      };
      loInput.addEventListener('input', update);
      hiInput.addEventListener('input', update);

      pair.appendChild(loInput);
      pair.appendChild(loOut);
      pair.appendChild(hiInput);
      pair.appendChild(hiOut);

      row.appendChild(label);
      row.appendChild(pair);
      block.appendChild(row);
    });

    container.appendChild(block);
  });
}

function makeRangeInput(min, max, value) {
  const input = document.createElement('input');
  input.type = 'range';
  input.min = min;
  input.max = max;
  input.step = 1;
  input.value = value;
  return input;
}

function renderLearnedColors() {
  const el = document.getElementById('learnedColors');
  const fmt = c => (c ? `h=[${c.h}] s=[${c.s}] v=[${c.v}]` : '(not learned yet)');
  el.textContent = `floor_color: ${fmt(config.floor_color)}\ntape_color: ${fmt(config.tape_color)}`;
}

function renderDistancePoints() {
  const el = document.getElementById('distancePoints');
  const points = config.distance_calibration_points;
  if (points.length === 0) {
    el.textContent = '(no calibration points yet)';
    return;
  }
  el.textContent = points
    .map(p => `distance=${p.distance_cm}cm  row_frac=${p.row_frac.toFixed(3)}  bbox_height_frac=${p.bbox_height_frac.toFixed(3)}`)
    .join('\n');
}

document.getElementById('captureDistanceButton').addEventListener('click', () => {
  const distance_cm = Number(document.getElementById('distanceInput').value);
  if (!distance_cm && distance_cm !== 0) {
    document.getElementById('distanceStatus').textContent = '✗ enter a distance first';
    return;
  }
  ui.send_message('capture_distance_point', { distance_cm });
});

document.getElementById('clearDistanceButton').addEventListener('click', () => {
  ui.send_message('clear_distance_calibration');
});

document.getElementById('saveConfigButton').addEventListener('click', () => {
  ui.send_message('save_config');
  document.getElementById('saveStatus').textContent = 'saved';
  setTimeout(() => (document.getElementById('saveStatus').textContent = ''), 2000);
});

document.getElementById('recordButton').addEventListener('click', () => {
  document.getElementById('recordStatus').textContent = 'recording...';
  ui.send_message('record_frame');
});

ui.send_message('list_recordings');
