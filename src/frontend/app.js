// app.js - simple front-end interactions (uses Fetch to call backend endpoints)
const uploadBtn = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');
const matNameInput = document.getElementById('mat-name');
const uploadResult = document.getElementById('upload-result');
uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  const name = matNameInput.value || 'material';
  if (!file) { uploadResult.innerText = 'Selecciona un archivo.'; return; }
  const fd = new FormData();
  fd.append('file', file);
  fd.append('material_name', name);
  const res = await fetch('/api/materials/upload', { method:'POST', body: fd });
  const j = await res.json();
  uploadResult.innerText = JSON.stringify(j, null, 2);
});

const calcBtn = document.getElementById('btn-calc');
const geomBox = document.getElementById('geom');
const calcResult = document.getElementById('calc-result');
calcBtn.addEventListener('click', async () => {
  let geom;
  try { geom = JSON.parse(geomBox.value); } catch(e) { calcResult.innerText = 'JSON inválido'; return; }
  const res = await fetch('/api/geometry/calc', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(geom)
  });
  const j = await res.json();
  if (res.ok) {
    calcResult.innerText = 'Cálculo completado. Se devolvieron ' + j.wl.length + ' puntos.';
    // show small plot (rudimentary)
    // for demo, print first 5 values
    calcResult.innerText += '\nR (first 5) = ' + j.R.slice(0,5).map(x=>x.toFixed(4)).join(', ');
  } else {
    calcResult.innerText = JSON.stringify(j);
  }
});

const fitBtn = document.getElementById('btn-fit');
const fitResult = document.getElementById('fit-result');
fitBtn.addEventListener('click', async () => {
  // demo trigger (server expects geometry and exp path)
  const payload = {
    geometry: {"layers":["air","film","substrate"], "thicknesses":[100], "angle_deg":0, "wavelength":{"start_nm":400,"end_nm":800,"points":401}},
    exp_path: "data/uploads/R_exp_demo.csv",
    initial_guess: [100.0]
  };
  const res = await fetch('/api/fit/run', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  const j = await res.json();
  fitResult.innerText = JSON.stringify(j, null, 2);
});
