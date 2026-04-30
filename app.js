// =============================================
//  CARDIOPREDICT — LÓGICA PRINCIPAL
//  Carga modelos ONNX y realiza predicciones
// =============================================

// ── Estado global ──
let scalerSession  = null;
let logregSession  = null;
let mlpSession     = null;
let activeModel    = 'logreg'; // 'logreg' | 'mlp'

// Columnas en el orden correcto
const FEATURE_COLS = [
  'age','sex','cp','trestbps','chol',
  'fbs','restecg','thalach','exang',
  'oldpeak','slope','ca','thal'
];

// ── Cargar modelos al iniciar ──
async function loadModels() {
  try {
    // Configurar ONNX Runtime
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';

    setStatus('loading', 'Cargando modelos…');

    // Cargar los 3 archivos ONNX desde la carpeta models/
    [scalerSession, logregSession, mlpSession] = await Promise.all([
      ort.InferenceSession.create('models/scaler.onnx'),
      ort.InferenceSession.create('models/logreg.onnx'),
      ort.InferenceSession.create('models/mlp.onnx'),
    ]);

    setStatus('ready', 'Modelos listos');
    console.log('✅ Modelos cargados correctamente');
  } catch (err) {
    setStatus('error', 'Error al cargar modelos');
    console.error('❌ Error cargando modelos ONNX:', err);
  }
}

function setStatus(type, text) {
  const dot  = document.querySelector('.status-dot');
  const span = document.getElementById('statusText');
  dot.className  = `status-dot ${type}`;
  span.textContent = text;
}

// ── Escalar features usando el scaler ONNX ──
async function scaleFeatures(rawArray) {
  const tensor = new ort.Tensor('float32', Float32Array.from(rawArray), [1, 13]);
  const result = await scalerSession.run({ float_input: tensor });
  // El scaler devuelve la salida escalada
  const outputKey = Object.keys(result)[0];
  return Array.from(result[outputKey].data);
}

// ── Predecir con el modelo activo ──
async function predictFromArray(rawFeatures) {
  if (!scalerSession || !logregSession || !mlpSession) {
    alert('Los modelos aún están cargando. Espera un momento e intenta de nuevo.');
    return null;
  }

  // Escalar
  const scaled = await scaleFeatures(rawFeatures);

  // Elegir sesión
  const session = activeModel === 'logreg' ? logregSession : mlpSession;

  // Crear tensor de entrada
  const tensor = new ort.Tensor('float32', Float32Array.from(scaled), [1, 13]);

  // Obtener nombre del input del modelo
  const inputName = session.inputNames[0];
  const result    = await session.run({ [inputName]: tensor });

  // La salida de clasificación sklearn→ONNX suele llamarse 'output_label' o 'label'
  const labelKey = session.outputNames.find(k => k.includes('label') || k === 'output_label') 
                   || session.outputNames[0];

  const prediction = Number(result[labelKey].data[0]);
  return prediction;
}

// ── PREDICCIÓN INDIVIDUAL ──
async function predictIndividual() {
  // Recoger valores del formulario
  const values = FEATURE_COLS.map(col => {
    const el  = document.getElementById(col);
    const val = parseFloat(el.value);
    return val;
  });

  // Validar que todos los campos estén llenos
  if (values.some(v => isNaN(v))) {
    alert('Por favor, completa todos los campos antes de predecir.');
    return;
  }

  const prediction = await predictFromArray(values);
  if (prediction === null) return;

  showIndividualResult(prediction);
}

function showIndividualResult(pred) {
  const card     = document.getElementById('individualResult');
  const icon     = document.getElementById('resultIcon');
  const title    = document.getElementById('resultTitle');
  const subtitle = document.getElementById('resultSubtitle');

  card.style.display = 'flex';
  card.className      = 'result-card';

  if (pred === 1) {
    card.classList.add('positive');
    icon.textContent    = '⚠️';
    title.textContent   = 'Enfermedad cardíaca detectada';
    subtitle.textContent = `El modelo de ${activeModel === 'logreg' ? 'Regresión Logística' : 'Red Neuronal'} predice presencia de enfermedad. Se recomienda evaluación médica especializada.`;
  } else {
    card.classList.add('negative');
    icon.textContent    = '✅';
    title.textContent   = 'Sin enfermedad cardíaca detectada';
    subtitle.textContent = `El modelo de ${activeModel === 'logreg' ? 'Regresión Logística' : 'Red Neuronal'} no detecta indicadores de enfermedad cardíaca.`;
  }

  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function clearForm() {
  FEATURE_COLS.forEach(col => {
    const el = document.getElementById(col);
    if (el.tagName === 'SELECT') el.value = '';
    else el.value = '';
  });
  document.getElementById('individualResult').style.display = 'none';
}

// ── PREDICCIÓN POR LOTES ──
async function handleCSV(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Cambiar aspecto del upload zone
  const zone = document.getElementById('uploadZone');
  zone.innerHTML = `
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="1.5">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
    <div class="upload-text">${file.name}</div>
    <div class="upload-hint">Procesando…</div>
  `;

  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: async (results) => {
      await processBatch(results.data);
      zone.innerHTML = `
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="1.5">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <div class="upload-text" style="color:#16a34a">${file.name} — Procesado</div>
        <div class="upload-hint">Haz clic para cargar otro archivo</div>
      `;
    },
    error: (err) => {
      alert('Error leyendo el CSV: ' + err.message);
    }
  });
}

async function processBatch(rows) {
  if (!scalerSession || !logregSession || !mlpSession) {
    alert('Los modelos aún están cargando.');
    return;
  }

  const predictions = [];
  const actuals     = [];

  for (const row of rows) {
    // Extraer features en el orden correcto
    const features = FEATURE_COLS.map(col => parseFloat(row[col]));

    // Verificar que la fila tenga datos válidos
    if (features.some(v => isNaN(v))) continue;

    const pred = await predictFromArray(features);
    if (pred === null) continue;

    predictions.push(pred);

    // El target real (puede ser 0/1 o 0–4)
    const realTarget = row['target'] !== undefined ? row['target'] : row['Target'];
    actuals.push(realTarget > 0 ? 1 : 0);
  }

  if (predictions.length === 0) {
    alert('No se encontraron filas válidas en el CSV. Verifica el formato.');
    return;
  }

  // Calcular métricas
  const metrics = calcMetrics(actuals, predictions);
  showBatchResults(actuals, predictions, metrics, rows);
}

function calcMetrics(actuals, preds) {
  let tp = 0, tn = 0, fp = 0, fn = 0;

  for (let i = 0; i < actuals.length; i++) {
    const a = actuals[i], p = preds[i];
    if (a === 1 && p === 1) tp++;
    else if (a === 0 && p === 0) tn++;
    else if (a === 0 && p === 1) fp++;
    else if (a === 1 && p === 0) fn++;
  }

  const accuracy  = (tp + tn) / (tp + tn + fp + fn);
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall    = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1        = precision + recall > 0
    ? 2 * (precision * recall) / (precision + recall) : 0;

  return { tp, tn, fp, fn, accuracy, precision, recall, f1 };
}

function showBatchResults(actuals, preds, m, rows) {
  document.getElementById('batchResults').style.display = 'block';

  // Métricas
  document.getElementById('bAccuracy').textContent  = (m.accuracy  * 100).toFixed(1) + '%';
  document.getElementById('bPrecision').textContent = (m.precision * 100).toFixed(1) + '%';
  document.getElementById('bRecall').textContent    = (m.recall    * 100).toFixed(1) + '%';
  document.getElementById('bF1').textContent        = (m.f1        * 100).toFixed(1) + '%';

  // Matriz de confusión
  document.getElementById('cmTN').textContent = m.tn;
  document.getElementById('cmFP').textContent = m.fp;
  document.getElementById('cmFN').textContent = m.fn;
  document.getElementById('cmTP').textContent = m.tp;

  // Tabla
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';
  document.getElementById('tableCount').textContent = `(${preds.length} pacientes)`;

  // Filtrar solo las filas válidas (matching con predictions)
  let validIdx = 0;
  for (let i = 0; i < rows.length && validIdx < preds.length; i++) {
    const row = rows[i];
    const features = FEATURE_COLS.map(col => parseFloat(row[col]));
    if (features.some(v => isNaN(v))) continue;

    const pred   = preds[validIdx];
    const actual = actuals[validIdx];
    const correct = pred === actual;
    validIdx++;

    const sex = row['sex'] == 1 ? 'M' : 'F';
    const tr  = document.createElement('tr');
    tr.innerHTML = `
      <td>${validIdx}</td>
      <td>${row['age']}</td>
      <td>${sex}</td>
      <td><span class="badge ${actual === 1 ? 'badge-positive' : 'badge-negative'}">${actual === 1 ? 'Enfermedad' : 'Sano'}</span></td>
      <td><span class="badge ${pred   === 1 ? 'badge-positive' : 'badge-negative'}">${pred   === 1 ? 'Enfermedad' : 'Sano'}</span></td>
      <td><span class="badge ${correct ? 'badge-correct' : 'badge-incorrect'}">${correct ? '✓ Correcto' : '✗ Incorrecto'}</span></td>
    `;
    tbody.appendChild(tr);
  }

  document.getElementById('batchResults').scrollIntoView({ behavior: 'smooth' });
}

// ── NAVEGACIÓN ──
function switchTab(tab) {
  // Actualizar nav
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  event.target.classList.add('active');

  // Mostrar/ocultar tabs
  document.getElementById('tab-individual').style.display = tab === 'individual' ? 'block' : 'none';
  document.getElementById('tab-batch').style.display      = tab === 'batch'      ? 'block' : 'none';
}

function selectModel(model, btn) {
  activeModel = model;
  document.querySelectorAll('.model-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  // Limpiar resultado previo
  document.getElementById('individualResult').style.display = 'none';
  document.getElementById('batchResults').style.display     = 'none';
}

// ── INICIO ──
window.addEventListener('DOMContentLoaded', loadModels);