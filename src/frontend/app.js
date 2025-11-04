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

// Run TMM using multipart/form-data: payload JSON + uploaded file(s)
const btnRunTMM = document.getElementById('btn-run-tmm');
const filmFileInput = document.getElementById('film-file-input'); // usa el id real

btnRunTMM.addEventListener('click', async () => {
  // Construye el modelo que enviarás al backend
  const model = {
    global: {
      angle: 70,
      wavelength_grid: { start_nm: 400, end_nm: 800, points: 401 }
    },
    layers: [
      { name: "air", modelType: "ambient" },
      // la capa 'film' espera un archivo: file_field debe coincidir con el nombre de campo que envías
      { name: "film", modelType: "file", thickness: 100, file_field: "film_file" },
      { name: "substrate", modelType: "glass" }
    ]
  };

  // Validaciones básicas
  if (!filmFileInput.files || filmFileInput.files.length === 0) {
    alert('Selecciona el archivo del material para la capa film antes de ejecutar TMM.');
    return;
  }

  const fd = new FormData();
  // colocamos el JSON como campo 'payload' (backend leerá request.form()['payload'])
  fd.append('payload', JSON.stringify(model));
  // añadimos el archivo con el nombre que coincide con file_field en el model
  fd.append('film_file', filmFileInput.files[0]); // el primer argumento es el nombre del campo

  // llamada fetch (no necesitas Content-Type, el navegador lo añade)
  try {
    const res = await fetch('/api/tmm/run', { method: 'POST', body: fd });
    const j = await res.json();
    if (!res.ok) {
      console.error('Error TMM:', j);
      alert('Error del servidor: ' + (j.detail || JSON.stringify(j)));
      return;
    }
    console.log('TMM result:', j);
    // ejemplo: mostrar psi/delta en consola o en la UI
    // convierto arrays para una gráfica simple o para debug
    // j.psi_deg  y j.delta_deg
    alert('TMM completado. Revisa la consola para resultados (psi/delta).');
  } catch (err) {
    console.error('Fetch error', err);
    alert('No se pudo conectar con el servidor. ¿Está corriendo el backend?');
  }
});

// Optimización: enviar modelo y archivos mediante FormData
document.getElementById('go-fit').addEventListener('click', async () => {
  try {
    const model = window.lastSavedModel; // asegúrate de tenerlo
    if (!model) {
      alert('No hay modelo guardado en window.lastSavedModel');
      return;
    }

    const fd = new FormData();
    // payload como campo string
    fd.append('payload', JSON.stringify(model));

    // Añade archivos que el usuario haya subido:
    // supongamos model.layers[i].file_field contiene el campo/filename que el servidor espera
    model.layers.forEach(layer => {
      if (layer.modelType === 'file' && layer.file_field && layer.fileObj) {
        // layer.fileObj debe ser el File elegido por el usuario (input.files[0])
        fd.append(layer.file_field, layer.fileObj, layer.fileObj.name);
      }
    });

    const res = await fetch('/api/tmm/run', {
      method: 'POST',
      body: fd
      // NO pongas Content-Type: fetch lo maneja automáticamente con Boundary
    });

    const j = await res.json();
    if (!res.ok) {
      alert('Error: ' + (j.detail || JSON.stringify(j)));
      return;
    }
    console.log('TMM result', j);
    alert('Optimización con archivos completada.');
  } catch (err) {
    console.error(err);
    alert('Error de conexión: ' + err.message);
  }
});

function showModelSummary(model) {
  // ... tu código anterior que construye la card ...
  card.innerHTML = `
    <div class="card-body">
      <h5 class="card-title">Modelo óptico (resumen)</h5>
      <p class="mb-1"><strong>Ángulo:</strong> ${model.global.angle}°, <strong>Polarización:</strong> ${model.global.polarization}</p>
      <p class="mb-1"><strong>Modo λ:</strong> ${model.global.wavelengthMode}${model.global.wavelengthMode==='range' ? ` (${model.global.wl_from}–${model.global.wl_to} nm, ${model.global.wl_steps} pasos)` : model.global.wavelengthMode==='single' ? ` (${model.global.wl_single} nm)` : ' (usar archivo)'} </p>
      <p class="mb-1"><strong>Capas:</strong> ${model.layers.length}</p>
      <ul>${model.layers.map((ly, i) => `<li>${i+1}. ${ly.name} — ${ly.thickness || ''} nm — ${ly.modelType} ${ly.optimize ? '(optimizable)' : ''}${ly.formula_latex ? ' — fórmula: ' + ly.formula_latex : ''}</li>`).join('')}</ul>
      <div class="mt-2">
        <button id="edit-model" class="btn btn-outline-secondary btn-sm">Editar modelo</button>
        <button id="go-fit" class="btn btn-success btn-sm ms-2">Ir a optimización</button>
      </div>
    </div>
  `;

  // exposición global para que otros handlers lo usen fácilmente
  window.lastSavedModel = model;

  // Insert card as before
  const topRow = document.querySelector('.container > .row');
  topRow.insertAdjacentElement('afterend', card);

  document.getElementById('edit-model').addEventListener('click', () => { document.getElementById('btn-next').click(); });

  // Nuevo handler: abrir optimización y enviar multipart/form-data con archivos
  document.getElementById('go-fit').addEventListener('click', async () => {
    try {
      const modelToSend = window.lastSavedModel;
      if (!modelToSend) {
        alert('No hay modelo guardado. Guarda el modelo antes de optimizar.');
        return;
      }

      // Construir FormData
      const fd = new FormData();
      fd.append('payload', JSON.stringify(modelToSend));

      // Añadir archivos: asumimos que durante la creación de layers
      // guardaste el File en model.layers[i].fileObj (ver nota abajo).
      for (const layer of modelToSend.layers) {
        if (layer.modelType === 'file') {
          // layer.file_field: nombre del campo que el backend espera (ej "film_file.csv")
          // layer.fileObj: el File object (input.files[0]) — debe haberse capturado al seleccionar el archivo
          if (!layer.file_field) {
            console.warn('Layer sin file_field:', layer);
            continue;
          }
          if (layer.fileObj) {
            fd.append(layer.file_field, layer.fileObj, layer.fileObj.name);
          } else {
            console.warn('No se encontró fileObj para layer', layer.name);
          }
        }
      }

      // Enviar a endpoint multipart que vamos a crear: /api/tmm/run-multipart
      const res = await fetch('/api/tmm/run-multipart', {
        method: 'POST',
        body: fd
      });

      const j = await res.json();
      if (!res.ok) {
        alert('Error del servidor: ' + (j.detail || JSON.stringify(j)));
        return;
      }

      // Aquí muestras/visualizas resultados — por ahora solo console.log
      console.log('TMM result', j);
      // Ejemplo: mostrar mensaje y refrescar gráficas
      alert('Optimización completada. Revisa la consola para resultados.');
      // Si quieres, llama a una función que grafique j.psi_deg/j.delta_deg
      // renderTMMResults(j);
    } catch (err) {
      console.error(err);
      alert('Error al llamar a la optimización: ' + err.message);
    }
  });
}

// wiring: después de definir fileRow, etc.
const fileInputElem = wrapper.querySelector('.layer-file');
if (fileInputElem) {
  fileInputElem.addEventListener('change', (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (file) {
      // Guardar nombre y referencia del File en el wrapper (DOM node)
      wrapper.dataset.fileName = file.name;
      wrapper._fileObj = file;
    } else {
      delete wrapper.dataset.fileName;
      wrapper._fileObj = null;
    }
  });
}
