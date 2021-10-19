let app = document.getElementById("typewriter");

let typewriter = new Typewriter(app, {
  loop: true,
  delay: 75,
});

typewriter
  .pauseFor(2500)
  .typeString("La Capital del Sol")
  .pauseFor(200)
  .deleteChars(10)
  .start();

$("#model_table_tr").append(`<td style="text-align:center">YOLOv5 Small</td>`);
$("#model_table_tr").append(`<td style="text-align:center">640x640 px</td>`);
$("#model_table_tr").append(`<td style="text-align:center">0.65</td>`);
$("#model_table_tr").append(`<td style="text-align:center">0.45</td>`);

let idx = 0;
$(`#results_table_tr_${idx}`).append(`<td style="text-align:center"></td>`);
$(`#results_table_tr_${idx}`).append(`<td style="text-align:center"></td>`);
$(`#results_table_tr_${idx}`).append(`<td style="text-align:center"></td>`);
$(`#results_table_tr_${idx}`).append(`<td style="text-align:center"></td>`);
$(`#results_table_tr_${idx}`).append(`<td style="text-align:center"></td>`);
$(`#results_table_tr_${idx}`).append(`<td style="text-align:center"></td>`);

const apiCaller = () => {
  let data = {};
  data.filename = $("#formFile").val();
  data.uploadedFile = $("#formFile").prop("files")[0];

  var form = new FormData();
  form.append("attachment", data["uploadedFile"], data["filename"]);

  var settings = {
    url: "http://localhost:5000/v1/img-object-detection/yolov5",
    method: "POST",
    timeout: 0,
    processData: false,
    mimeType: "multipart/form-data",
    contentType: false,
    data: form,
  };

  const submit_btn = document.getElementById("submit_btn");
  if (data.filename) {
    submit_btn.disabled = true;
    $("#submit_btn").empty();
    $("#submit_btn").wrapInner(
      '<span id="loading_spinner" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>\nLoading...'
    );
  }

  $.ajax(settings).done(function (response) {
    submit_btn.disabled = false;
    $("#submit_btn").empty();
    $("#submit_btn").wrapInner("Submit");

    console.log(response);
    // console.log(status);
    const responseJson = JSON.parse(response);
    console.log(responseJson);

    // TODO: Trye to make  status code work. Add style to detected img. Show logs info
    // if (status == 200) {

    // TODO: agregar logica para descomprimir el attachment recibido con las deteccion y armar carrusel
    // $("#image-from-server").attr(
    //   "src",
    //   "data:image/png;base64," + responseJson.inference
    // );

    // saveAs(responseJson.inference, "inference.zip");

    // var zip = new JSZip();
    // zip.loadAsync(responseJson.inference, { base64: true });
    // zip.generateAsync({ type: "base64" }).then(function (content) {
    //   location.href = "data:application/zip;base64," + content;
    // });
    // location.href = responseJson.inference;
    // var zip = new JSZip();
    // zip.loadAsync(responseJson.inference, { base64: true })
    // zip.generateAsync({ type: "base64" }).then(function (content) {
    //     saveAs(content, "inference.zip");
    //   });

    // saveAs(responseJson.inference, "inference.zip");
    // JSZipUtils.getBinaryContent(responseJson.inference, function (err, data) {
    //   if (err) {
    //     throw err; // or handle err
    //   }

    //   JSZip.loadAsync(data).then(function () {
    //     saveAs(data, "inference.zip");
    //     console.log("TERMINE");
    //   });
    // });

    const input_img_size = responseJson.logs.input_img_size;
    const conf_thres = responseJson.logs.conf_thres;
    const iou_thres = responseJson.logs.iou_thres;
    const total_time = responseJson.logs.total_time;
    const raw_logs = responseJson.logs.raw_logs;

    // TODO: agregar una row por cada elemento
    for (inference of responseJson.logs.inference_logs) {
      idx = idx + 1;
      const pivots = inference.detections.pivots
        ? inference.detections.pivots
        : 0;
      const silobolsas = inference.detections.silobolsas
        ? inference.detections.silobolsas
        : 0;
      const img_size = inference.img_size;
      const inference_time = inference.inference_time;
      const image_name = inference.image_name;

      // const results_table = document.getElementById("results_table");
      // results_table.deleteRow(1);

      const model_table = document.getElementById("model_table");
      model_table.deleteRow(1);

      $("#results_table").append(`<tr id="results_table_tr_${idx}"></tr>`);

      $(`#results_table_tr_${idx}`).append(
        `<td style="text-align:center">${image_name}</td>`
      );
      $(`#results_table_tr_${idx}`).append(
        `<td style="text-align:center">${total_time}</td>`
      );
      $(`#results_table_tr_${idx}`).append(
        `<td style="text-align:center">${inference_time}</td>`
      );
      $(`#results_table_tr_${idx}`).append(
        `<td style="text-align:center">${img_size}</td>`
      );
      $(`#results_table_tr_${idx}`).append(
        `<td style="text-align:center">${pivots}</td>`
      );
      $(`#results_table_tr_${idx}`).append(
        `<td style="text-align:center">${silobolsas}</td>`
      );

      $("#model_table").append(`<tr id="model_table_tr"></tr>`);
      $("#model_table_tr").append(
        `<td style="text-align:center">YOLOv5 Small</td>`
      );
      $("#model_table_tr").append(
        `<td style="text-align:center">${input_img_size}</td>`
      );
      $("#model_table_tr").append(
        `<td style="text-align:center">${conf_thres}</td>`
      );
      $("#model_table_tr").append(
        `<td style="text-align:center">${iou_thres}</td>`
      );
    }

    // }
  });
};
