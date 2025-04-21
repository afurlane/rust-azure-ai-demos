use std::fs;
use image::GenericImageView;
use reqwest::Client;
use serde_json::Value;

const AZURE_ENDPOINT: &str = "https://img-computer-vision.cognitiveservices.azure.com/";
const API_KEY: &str = "4YzGv5y5WlTTmTGdWmrVwJrlCxOzr1uo0z0pPZD779rLtP4u3heWJQQJ99BDAC5RqLJXJ3w3AAAFACOGEYYO";

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
async fn detect_license_plate_custom(image_path: &str) -> Result<(u32, u32, u32, u32), Box<dyn std::error::Error>> {

    let img_bytes = std::fs::read(image_path)?;

    let client = Client::new();
    let res = client.post(format!("{}/predict", AZURE_ENDPOINT))
        .header("Prediction-Key", API_KEY)
        .header("Content-Type", "application/octet-stream")
        .body(img_bytes)
        .send().await.expect("Errore nella richiesta");

    let json: Value = res.json().await?;
    
    if let Some(predictions) = json["predictions"].as_array() {
        for pred in predictions {
            if pred["tagName"].as_str() == Some("license_plate") {
                let rect = &pred["boundingBox"];
                let img_width = json["image"]["width"].as_u64().unwrap() as u32;
                let img_height = json["image"]["height"].as_u64().unwrap() as u32;
                return Ok((
                    (rect["left"].as_f64().unwrap() * img_width as f64) as u32,
                    (rect["top"].as_f64().unwrap() * img_height as f64) as u32,
                    (rect["width"].as_f64().unwrap() * img_width as f64) as u32,
                    (rect["height"].as_f64().unwrap() * img_height as f64) as u32,
                ));
            }
        }
    }

    Err("Nessuna targa trovata".into())
}

async fn detect_license_plate(image_path: &str) -> Result<(u32, u32, u32, u32), Box<dyn std::error::Error>> {

    let image_bytes = std::fs::read(image_path)?;
    
    let client = Client::new();
    let res = client.post(format!("{}/vision/v3.0/analyze?visualFeatures=Objects", AZURE_ENDPOINT))
        .header("Ocp-Apim-Subscription-Key", API_KEY)
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

async fn crop_license_plate(image_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (x, y, width, height) = detect_license_plate(image_path).await?;
    
    let img = image::open(image_path)?;
    let cropped = img.view(x, y, width, height).to_image();
    
    cropped.save(output_path)?;
    
    Ok(())
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
//    let image_url = "https://example.com/targa.jpg"; // URL dell'immagine da analizzare
    let client = Client::new();

/*
    let response = client.post(format!("{}/vision/v3.2/ocr", AZURE_ENDPOINT))
        .header("Ocp-Apim-Subscription-Key", API_KEY)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({"url": image_url}))
        .send()
        .await?;
*/
    // let image_data = fs::read("downloaded_image.jpg").expect("Errore nella lettura del file");
    // let image_data_bytes = include_bytes!("IMG_6704 2025-04-21 09_15_36.JPG");
    const IMAGE_PATH: &str = "/home/afurlane/projects/rust-azure-ai-demos/src/IMG_6482 2025-04-21 09_15_57_2.JPG";
    crop_license_plate(IMAGE_PATH, "./test_img.jpg").await.unwrap();
    let image_data_bytes = fs::read("test_img.jpg").expect("Errore nella lettura del file");
    let image_data = image_data_bytes.to_vec();

    let response = client.post(format!("{}/vision/v3.2/ocr", AZURE_ENDPOINT))
        .header("Ocp-Apim-Subscription-Key", API_KEY)
        .header("Content-Type", "application/octet-stream")
        .body(image_data)
        .send()
        .await.unwrap();

    let json: Value = response.json().await?;
    
    println!("Json {:?}", json);
    if let Some(regions) = json["regions"].as_array() {
        println!("Regions {:?}", regions);
        for region in regions {
            if let Some(lines) = region["lines"].as_array() {
                for line in lines {
                    if let Some(words) = line["words"].as_array() {
                        let targa: String = words.iter().map(|w| w["text"].as_str().unwrap_or("")).collect::<Vec<_>>().join(" ");
                        println!("Numero di targa: {}", targa);
                    }
                }
            }
        }
    }

    Ok(())
}
