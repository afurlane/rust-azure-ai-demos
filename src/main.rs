use std::{error::Error, fs, path::Path, time::Instant};
use image::{DynamicImage, GenericImageView, ImageReader};
use ort::{tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder, Value};
use reqwest::Client;
use serde_derive::{Deserialize, Serialize};
use ndarray::{Array, CowArray, IxDyn};
use dotenv::dotenv;

// Per questi vedi: https://learn.microsoft.com/it-it/azure/ai-services/Custom-Vision-Service/quickstarts/image-classification?tabs=windows%2Cvisual-studio&pivots=programming-language-rest-api
// Percorso del modello scaricato
const MODEL_PATH: &str = "./src/onnx_model/model.onnx";

const AZURE_API_CUSTOM_VISION_INDIVIDUA_TARGHE_BODY_ENDPOINT: &str = "https://cvisionsvctest-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/9421e0dd-b5c5-4133-bd49-4ffbc6b9549b/detect/iterations/individua-targhe/image";
const AZURE_API_CUSTOM_VISION_INDIVIDUA_TARGHE_BODY_KEY: &str = "<keys>";
const AZURE_API_CUSTOM_VISION_INDIVIDUA_TARGHE_BODY_HEADER: &str = "Prediction-Key";

// Vedi: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/call-analyze-image-40?pivots=programming-language-rest-api
// e https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/overview-ocr
const AZURE_API_VISION_INDIVIDUA_TESTO_TARGHE_BODY_ENDPOINT: &str = "https://visionsvctest.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read,caption,denseCaptions,smartCrops";
const AZURE_API_VISION_INDIVIDUA_TESTO_TARGHE_BODY_KEY: &str = "<keys>";
const AZURE_API_VISION_INDIVIDUA_TESTO_TARGHE_BODY_HEADER: &str = "Ocp-Apim-Subscription-Key";

const TEST_IMAGE_PATH: &str = "../ai-img-test/test_img.jpg";
const IMAGE_PATH: &str = "../ai-img-test/IMG_6704 2025-04-21 09_15_36_2.JPG";

// Vedi questo readme per i limiti associati a determinati account (demo e prod)
// https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/custom-vision-service/limits-and-quotas.md

// Con modello allenato appositamente sulle targhe (servono almeno 200 immagini di targhe per avere una buona accuratezza)
/*
🔹 1️⃣ Creare un progetto su Azure Custom Vision
Vai su Azure Custom Vision.

Accedi con il tuo account Azure e crea un nuovo progetto.

Imposta il tipo di progetto come "Object Detection", perché vuoi individuare targhe all'interno delle immagini.

🔹 2️⃣ Caricare immagini delle targhe
Raccogli immagini di veicoli con targhe visibili da diverse angolazioni e condizioni.

Carica le immagini nel progetto Custom Vision.

Etichetta manualmente le targhe selezionando l'area dove compaiono e assegnando un tag come "license_plate".

🔹 3️⃣ Addestrare il modello
Una volta che hai abbastanza immagini etichettate (idealmente almeno 50-100 targhe per buoni risultati), avvia il processo di training.

Azure Custom Vision utilizzerà AI per imparare a individuare targhe nelle immagini.

Dopo l'addestramento, il modello ti fornirà un'accuratezza (%). Se necessario, puoi caricare altre immagini e rifare il training.

🔹 4️⃣ Pubblicare il modello e ottenere un API endpoint
Dopo aver ottenuto un modello con buona accuratezza, pubblicalo.

Otterrai un URL dell'API e una chiave API, che potrai usare da Rust per analizzare nuove immagini e trovare targhe automaticamente!
*/

//
// API Custom Vision predict reader result
//

// Mappiamo anche la risposta dell'API
// Example code that deserializes and serializes the model.
// extern crate serde;
// #[macro_use]
// extern crate serde_derive;
// extern crate serde_json;
//
// use generated_module::predict;
//
// fn main() {
//     let json = r#"{"answer": 42}"#;
//     let model: predict = serde_json::from_str(&json).unwrap();
// }

//
// Merito di: https://app.quicktype.io/
//
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Predict {
    created: String,
    id: String,
    iteration: String,
    predictions: Vec<Prediction>,
    project: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Prediction {
    bounding_box: BoundingBox,
    probability: f64,
    tag_id: String,
    tag_name: TagName,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    height: f64,
    left: f64,
    top: f64,
    width: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagName {
    Targa,
}

//
// API Text reader result
//

// Example code that deserializes and serializes the model.
// extern crate serde;
// #[macro_use]
// extern crate serde_derive;
// extern crate serde_json;
//
// use generated_module::textReadResult;
//
// fn main() {
//     let json = r#"{"answer": 42}"#;
//     let model: textReadResult = serde_json::from_str(&json).unwrap();
// }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextReadResult {
    caption_result: Option<CaptionResult>,
    dense_captions_result: Option<DenseCaptionsResult>,
    metadata: Option<Metadata>,
    model_version: String,
    read_result: ReadResult,
    smart_crops_result: SmartCropsResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionResult {
    confidence: f64,
    text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseCaptionsResult {
    values: Vec<DenseCaptionsResultValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DenseCaptionsResultValue {
    #[serde(rename = "boundingBox")]
    bounding_box: ImageBoundingBox,
    confidence: f64,
    text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageBoundingBox {
    h: i64,
    w: i64,
    x: i64,
    y: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    height: i64,
    width: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadResult {
    blocks: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    lines: Vec<Line>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Line {
    bounding_polygon: Vec<BoundingPolygon>,
    text: String,
    words: Vec<Word>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingPolygon {
    x: i64,
    y: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Word {
    bounding_polygon: Vec<BoundingPolygon>,
    confidence: f64,
    text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmartCropsResult {
    values: Vec<SmartCropsResultValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmartCropsResultValue {
    aspect_ratio: f64,
    bounding_box: ImageBoundingBox,
}

//
// Actual code
// 
async fn detect_license_plate_custom(image_path: &str) -> Result<(u32, u32, u32, u32), Box<dyn std::error::Error>> {

    let path = Path::new(image_path);
    let reader = ImageReader::open(path)?;
    let dimensions = reader.into_dimensions()?;

    let img_bytes = std::fs::read(image_path)?;
    let client = Client::new();
    let start = Instant::now();
    let res = client.post(AZURE_API_CUSTOM_VISION_INDIVIDUA_TARGHE_BODY_ENDPOINT)
        .header(AZURE_API_CUSTOM_VISION_INDIVIDUA_TARGHE_BODY_HEADER, AZURE_API_CUSTOM_VISION_INDIVIDUA_TARGHE_BODY_KEY)
        .header("Content-Type", "application/octet-stream")
        .body(img_bytes)
        .send().await.expect("Errore nella richiesta");
    
    let duration = start.elapsed();
    println!("Durata chiamata identificazione oggetto: {:?}", duration);

    let predict: Predict = res.json::<Predict>().await?;
    for pred in predict.predictions {
        if pred.tag_name == TagName::Targa && pred.probability >= 0.9 {
            let rect = &pred.bounding_box ;
            let img_width = dimensions.0;
            let img_height = dimensions.1;
            return Ok((
                (rect.left * img_width as f64) as u32,
                (rect.top  * img_height as f64) as u32,
                (rect.width * img_width as f64) as u32,
                (rect.height * img_height as f64) as u32,
            ));
        }
    }

    Err("Nessuna targa trovata".into())
}


async fn crop_license_plate(image_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // let (x, y, width, height) = detect_license_plate(image_path).await?;
    let (x, y, width, height) = detect_license_plate_custom(image_path).await?;

    let img = image::open(image_path)?.into_rgb8();
    let cropped = img.view(x, y, width, height);
    let save_image = DynamicImage::ImageRgb8(cropped.to_image());
    save_image.save(output_path)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // let image_url = "https://example.com/targa.jpg"; // URL dell'immagine da analizzare
    let client = Client::new();

    // let image_data = fs::read("downloaded_image.jpg").expect("Errore nella lettura del file");
    // let image_data_bytes = include_bytes!("IMG_6704 2025-04-21 09_15_36.JPG");
    crop_license_plate(IMAGE_PATH, TEST_IMAGE_PATH).await.unwrap();
    let image_data_bytes = fs::read(TEST_IMAGE_PATH).expect("Errore nella lettura del file");
    let image_data = image_data_bytes.to_vec();

    let start = Instant::now();
    // let response = client.post(format!("{}/vision/v3.2/ocr", AZURE_AI_ENDPOINT))
    let response = client.post(AZURE_API_VISION_INDIVIDUA_TESTO_TARGHE_BODY_ENDPOINT)
        .header(AZURE_API_VISION_INDIVIDUA_TESTO_TARGHE_BODY_HEADER, AZURE_API_VISION_INDIVIDUA_TESTO_TARGHE_BODY_KEY)
        .header("Content-Type", "application/octet-stream")
        .body(image_data)
        .send()
        .await.unwrap();
    
    let duration = start.elapsed();
    println!("Durata chiamata riconoscimento: {:?}", duration);

    let read_result: TextReadResult = response.json::<TextReadResult>().await?;
    
    for block in read_result.read_result.blocks {
        for line in block.lines {            
            println!("Numero di targa: {}", line.text);
        }
    }

    test_detect_onnx(TEST_IMAGE_PATH).await?;

    Ok(())
}

async fn test_detect_onnx(image_path: &str)  -> Result<(), Box<dyn Error>> {

    // Creazione dell'ambiente ONNX Runtime
    let environment = Environment::builder()
        .with_name("object-detection")
        .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
        .build()?;

    // Creazione della sessione per il modello
    let session = SessionBuilder::new(&environment.into_arc())?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(MODEL_PATH)?;

    // Creazione dell'allocatore
    let allocator = session.allocator();

    // Caricamento dell'immagine
    // let image_path = "path_all'immagine.jpg";
    let img = image::open(image_path)?;

    // Visualizzazione delle dimensioni dell'immagine
    println!("Dimensioni immagine: {}x{}", img.width(), img.height());

    // Preprocessing: ridimensionamento e normalizzazione
    let resized_img = img.resize_exact(320, 320, image::imageops::FilterType::CatmullRom);
    let mut input_tensor: Vec<f32> = Vec::new();

    for pixel in resized_img.to_rgb8().pixels() {
        input_tensor.push(pixel[0] as f32); // Normalizzazione R
        input_tensor.push(pixel[1] as f32); // Normalizzazione G
        input_tensor.push(pixel[2] as f32 ); // Normalizzazione B
    }

    // Definizione della forma del tensore
    let input_tensor_shape = IxDyn(&[1, 3, 320, 320]);

    // Conversione in un array compatibile utilizzando CowArray
    let input_array = CowArray::from(Array::from_shape_vec(input_tensor_shape, input_tensor)?);

    // Creazione del valore compatibile con ort
    let input_value = Value::from_array(allocator, &input_array)?;

    let start = Instant::now();
    // Esecuzione della predizione
    let outputs = session.run(vec![input_value])?;
    let duration = start.elapsed();
    println!("Durata chiamata di identificazione dell'oggetto con ORT-ONNX (CUDA): {:?}", duration);

    // Interpretazione dell'output: estrazione dei dati da CppOwned
    // Ora guardare il readme del modello!
    let mut boxes: Option<OrtOwnedTensor<f32, _>> = None;
    let mut scores: Option<OrtOwnedTensor<f32, _>> = None;
    let mut classes: Option<OrtOwnedTensor<i64, _>> = None;

    for (index, output) in outputs.iter().enumerate() {
        if index == 0 && output.is_tensor()? {
            // Estrarre i bounding boxes
            boxes = Some(output.try_extract()?);
            println!("Bounding Boxes: {:?}", boxes);
        } else if index == 2 && output.is_tensor()? {
            // Estrarre i confidence scores
            scores = Some(output.try_extract()?);
            println!("Scores: {:?}", scores);
        } else if index == 1 && output.is_tensor()? {
            // Estrarre le classi rilevate
            classes = Some(output.try_extract()?);
            println!("Classes: {:?}", classes);
        }
    }

    if boxes.is_some() && scores.is_some() && classes.is_some() {
        let img_width = img.width() as f32;
        let img_height = img.height() as f32;
        let boxxes = boxes.unwrap();
        let scores_view = scores.as_ref().unwrap().view(); // Ottieni una vista ndarray
        println!("Shape dei punteggi: {:?}", scores_view.shape()); // Controlla la forma del tensore
        if let Some((max_index, max_score)) = scores_view.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
            println!("Massimo punteggio: {:.2} (indice: {})", max_score, max_index);
            let boxes_view = boxxes.view();            
            // let scorre = score.view().rows().into_iter().last();
            // print!("Score {:?}", score);
            let x_min = (boxes_view[[0, max_index, 0]] * img_width).max(0.0).min(img_width).round() as u32;
            let y_min = (boxes_view[[0, max_index, 1]] * img_height).max(0.0).min(img_height).round() as u32;
            let x_max = (boxes_view[[0, max_index, 2]] * img_width).max(0.0).min(img_width).round() as u32;
            let y_max = (boxes_view[[0, max_index, 3]] * img_height).max(0.0).min(img_height).round() as u32;
    
            println!(
                "Classe: {}, Punteggio: {:.2}, Bounding Box: ({}, {}, {}, {})",
                "targa", max_score, x_min, y_min, x_max, y_max
            );

            let img_to_crop = img.clone().into_rgb8();
            let cropped = img_to_crop.view(x_min, y_min, x_max - x_min, y_max - y_min);
            let save_image = DynamicImage::ImageRgb8(cropped.to_image());
            save_image.save(format!("targa_{}_{}.jpg", x_min, y_min))?;        
        }
    }
    Ok({})
}
/*
async fn detect_license_plate(image_path: &str) -> Result<(u32, u32, u32, u32), Box<dyn std::error::Error>> {

    let image_bytes = std::fs::read(image_path)?;
    
    let client = Client::new();
    let res = client.post(format!("{}/vision/v3.0/analyze?visualFeatures=Objects", AZURE_AI_ENDPOINT))
        .header("Ocp-Apim-Subscription-Key", AZURE_AI_API_KEY)
        .header("Content-Type", "application/octet-stream")
        .body(image_bytes)
        .send().await.unwrap();

    
    let json: Value = res.json().await?;
    println!("detect_license_plate: {}", json);
    
    // Cerca un oggetto identificato come "license plate"
    if let Some(objects) = json["objects"].as_array() {
        println!("detect_license_plate: {:?}", objects);
        for obj in objects {
            if obj["object"].as_str() == Some("license plate") {
                let rect = &obj["rectangle"];
                return Ok((
                    rect["x"].as_u64().unwrap() as u32,
                    rect["y"].as_u64().unwrap() as u32,
                    rect["w"].as_u64().unwrap() as u32,
                    rect["h"].as_u64().unwrap() as u32,
                ));
            }
        }
    }

    Err("Nessuna targa trovata".into())
}
*/
