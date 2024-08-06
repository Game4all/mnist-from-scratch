// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

var canvas = new fabric.Canvas('user-canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 28;
canvas.freeDrawingBrush.color = "#000000";

// Canvas the user can draw on
const srcCanvas = document.getElementById("user-canvas");
const srcCanvasCtx = srcCanvas.getContext("2d");

// Offscreen canvas used for scaling the drawing to a 28x28 pixel image.
const croppedCanvas = new OffscreenCanvas(28, 28);
const croppedCanvasCtx = croppedCanvas.getContext('2d');

// Offscreen canvas used for centering the drawing before scaling it down.
const centeredCanvas = new OffscreenCanvas(srcCanvas.width, srcCanvas.height);
const centeredCanvasCtx = centeredCanvas.getContext("2d");

// Canvas for drawing the results probability distribution
const resultCanvas = document.getElementById("results");
const resultCanvasCtx = resultCanvas.getContext("2d");


/// Compute a hitbox from the drawing in the specified canvas
/// This hitbox will be then used to center the drawing in the canvas.
function computeHitbox(ctx, width, height) {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    let minX = width, minY = height, maxX = 0, maxY = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const alpha = data[(y * width + x) * 4 + 3];
            if (alpha > 0) {
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }
    }

    return { minX, minY, maxX, maxY };
}


/// Centers the image according to the previously computed hitboxs
function centerImage(hitbox) {
    const hitboxWidth = hitbox.maxX - hitbox.minX;
    const hitboxHeight = hitbox.maxY - hitbox.minY;

    const imageData = srcCanvasCtx.getImageData(hitbox.minX, hitbox.minY, hitboxWidth, hitboxHeight);

    centeredCanvasCtx.clearRect(0, 0, centeredCanvas.width, centeredCanvas.height);
    centeredCanvasCtx.putImageData(imageData, (centeredCanvas.width - hitboxWidth) / 2, (centeredCanvas.height - hitboxHeight) / 2);
}

/// Crops the canvas in an offscreen canvas, centers the contents and stores the data for in the input tensor memory.
function centerAndCropImage() {
    const hitbox = computeHitbox(srcCanvasCtx, srcCanvas.width, srcCanvas.height);
    centerImage(hitbox);

    croppedCanvasCtx.clearRect(0, 0, croppedCanvas.width, croppedCanvas.height);
    croppedCanvasCtx.drawImage(centeredCanvas, 0, 0, centeredCanvas.width, centeredCanvas.height, 0, 0, croppedCanvas.width, croppedCanvas.height);

    // convert data to grayscale
    const imgData = croppedCanvasCtx.getImageData(0, 0, croppedCanvas.width, croppedCanvas.height, {});
    var grayScaleImgData = new Float32Array(imgData.width * imgData.height)
    for (var i = 0; i < imgData.data.length; i += 4) {
        var a = imgData.data[i + 3];
        grayScaleImgData[i / 4] = a / 256.0;
    }

    return grayScaleImgData;
}

/// Draws the logit index on top of the distribution
function drawLogitNumbers() {
    resultCanvasCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    const barWidth = resultCanvas.width / 10 - 10;
    for (let i = 0; i < 10; i++) {
        const text = i.toString();
        const textWidth = resultCanvasCtx.measureText(text).width;
        const textX = (resultCanvas.width / 10) * i + 5 + (barWidth / 2) - (textWidth / 2);

        resultCanvasCtx.fillStyle = 'black';
        resultCanvasCtx.fillText(text, textX, 10);
    }
}

/// Draws the output logits score distribution
function drawLogitScores(result, logits) {
    drawLogitNumbers();
    const barWidth = resultCanvas.width / 10 - 10;
    for (let i = 0; i < logits.length; i++) {
        resultCanvasCtx.fillStyle = 'cyan';
        resultCanvasCtx.fillRect(
            (resultCanvas.width / 10) * i + 5,
            resultCanvas.height - 20,
            barWidth,
            -logits[i] * 50
        );
    }

    document.getElementById("pred-class").textContent = `Prediction : ${result} (${(logits[result] * 100.0).toFixed(2)}%)`;
}

/// Runs model inference and prints the logit distribution.
function guess() {
    const binaryData = centerAndCropImage();

    const inputTensor = new Float32Array(model.instance.exports.memory.buffer, model.instance.exports.getInputDataPointer(), binaryData.length);
    inputTensor.set(binaryData);

    const result = model.instance.exports.runInference();
    const logits = new Float32Array(model.instance.exports.memory.buffer, model.instance.exports.getOutputLogitsPointer(), 10);

    drawLogitScores(result, logits);
}


const model = await WebAssembly.instantiateStreaming(fetch("webmnist.wasm"));
console.log("Model initialization returned : " + (model.instance.exports.init() != 0 ? "an error" : "OK"));

drawLogitNumbers();

// clear button
document.getElementById("clear-btn").addEventListener("click", (_, __) => {
    canvas.clear();
    document.getElementById("pred-class").textContent = "Prediction : ???";
    drawLogitNumbers();
});

// correct button
document.getElementById("correct-btn").addEventListener("click", () => {
    while (true) {
        const wInput = window.prompt("Specify the correct digit label", "0");

        if (wInput === null) return;

        const targetValue = parseInt(wInput);

        if (isNaN(targetValue) || targetValue < 0 || targetValue >= 10) {
            alert("An invalid digit label was specified!");
        } else {
            model.instance.exports.train(targetValue);
            guess();
            return;
        }
    }
});

canvas.on("mouse:up", _ => {
    canvas.renderAll();
    guess();
});