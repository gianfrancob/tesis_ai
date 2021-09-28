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

const apiCaller = () => {
  console.log("holaaaa");
  console.log("filename: ", $("#formFile").val());

  let data = {};
  data["filename"] = $("#formFile").val();
  data["uploadedFile"] = $("#formFile").prop("files")[0];

  var form = new FormData();
  form.append("image", data["uploadedFile"], data["filename"]);

  var settings = {
    url: "http://localhost:5000/v1/img-object-detection/yolov5",
    method: "POST",
    timeout: 0,
    processData: false,
    mimeType: "multipart/form-data",
    contentType: false,
    data: form,
  };

  $.ajax(settings).done(function (response) {
    console.log(response);
    // console.log(status);
    const responseJson = JSON.parse(response);
    console.log(responseJson);

    // TODO: Trye to make  status code work. Add style to detected img. Show logs info
    // if (status == 200) {
    $("#image-from-server").attr(
      "src",
      "data:image/png;base64," + responseJson.image
    );
    $("#image_name").attr("src", responseJson.logs.image_name);
    $("#pivots").attr("src", responseJson.logs.detections.pivots);
    $("#silobolsas").attr("src", responseJson.logs.detections.silobolsas);
    $("#input_img_size").attr("src", responseJson.logs.input_img_size);
    $("#img_size").attr("src", responseJson.logs.img_size);
    $("#conf_thres").attr("src", responseJson.logs.conf_thres);
    $("#iou_thres").attr("src", responseJson.logs.iou_thres);
    $("#inference_time").attr("src", responseJson.logs.inference_time);
    $("#total_time").attr("src", responseJson.logs.total_time);
    $("#raw_logs").attr("src", responseJson.logs.raw_logs);

    // }
  });

  // $("#image-from-server").attr("src", "data:image/;base64," + data["response"]);
  // $("#image-from-server").attr("src", data["response"]);

  // todo: add root input element
  // data["root"] = "POST"
  //console.log(data)
  //var body = JSON.stringify(data)
  //$.ajax({
  //url: "https://us-central1-pycatj.cloudfunctions.net/pycatjify",
  // contentType: "application/json",
  // data: body,
  // dataType: "json",
  // type: 'POST',
  // success: function (response) {
  // $('#out_form').val(response.data)
  // }
  // });
};
