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
    data["response"] = response;
    $("#image-from-server").attr(
      "src",
      "data:image/;base64," +
        btoa(unescape(encodeURIComponent(data["response"])))
    );
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
