import SwiftUI
import CoreML
import AVFoundation
import VideoToolbox

// ドメインリスト (変更なし)
let domains = [
    "bibimpbap", "chahan", "chikenrice", "curry", "ebichill",
    "gratin", "gyudon", "hiyachu", "kaisendon", "katsudon",
    "meatspa", "omelet", "omurice", "oyakodon", "pilaf",
    "pizza", "ramen", "rice", "soba", "steak",
    "tendon", "unadon", "yakisoba"
]
// UIImage拡張機能
extension UIImage {
    // アスペクト比を無視してリサイズ (今回は使用しない可能性あり)
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        defer { UIGraphicsEndImageContext() }
        self.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    // --- 追加: アスペクト比を維持してリサイズ＆中央クロップ ---
    func resizeAspectFill(to targetSize: CGSize) -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        
        let imageSize = self.size
        let targetWidth = targetSize.width
        let targetHeight = targetSize.height
        
        // アスペクト比を計算
        let widthRatio = targetWidth / imageSize.width
        let heightRatio = targetHeight / imageSize.height
        
        // Fill するためのスケールを選択 (大きい方の比率を使う)
        let scaleFactor = max(widthRatio, heightRatio)
        
        // スケーリング後のサイズを計算
        let scaledWidth = imageSize.width * scaleFactor
        let scaledHeight = imageSize.height * scaleFactor
        
        // 描画する領域（中央揃え）を計算
        let drawingRect = CGRect(
            x: (targetWidth - scaledWidth) / 2.0,
            y: (targetHeight - scaledHeight) / 2.0,
            width: scaledWidth,
            height: scaledHeight
        )
        
        // 描画コンテキストを作成
        UIGraphicsBeginImageContextWithOptions(targetSize, false, self.scale) // scaleを考慮
        defer { UIGraphicsEndImageContext() }
        
        // 画像を描画
        self.draw(in: drawingRect)
        
        // 新しいUIImageを取得
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        return newImage
    }
    
    func rotateLeft() -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        return UIImage(cgImage: cgImage, scale: self.scale, orientation: .left)
    }
    
    func rotateRight() -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        return UIImage(cgImage: cgImage, scale: self.scale, orientation: .right)
    }
    
    // UIImageからCVPixelBufferへの変換関数 (ログ追加)
    func toCVPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(size.width),
                                         Int(size.height),
                                         kCVPixelFormatType_32ARGB, // Use ARGB for context compatibility
                                         attributes,
                                         &pixelBuffer)
        
        guard status == kCVReturnSuccess, let unwrappedPixelBuffer = pixelBuffer else {
            print("Error [toCVPixelBuffer]: Failed to create pixel buffer. Status: \(status)") // 詳細ログ
            return nil
        }
        
        CVPixelBufferLockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let pixelData = CVPixelBufferGetBaseAddress(unwrappedPixelBuffer) else {
            print("Error [toCVPixelBuffer]: Failed to get base address of pixel buffer.")
            return nil
        }
        print("[toCVPixelBuffer]: Pixel buffer base address obtained.") // 成功ログ
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        // Use CGImageAlphaInfo.noneSkipFirst for ARGB format
        let bitmapInfo = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue // Use Little Endian for iOS typical context
        
        guard let context = CGContext(data: pixelData,
                                      width: Int(size.width),
                                      height: Int(size.height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(unwrappedPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: bitmapInfo)
        else {
            print("Error [toCVPixelBuffer]: Failed to create CGContext.")
            return nil
        }
        print("[toCVPixelBuffer]: CGContext created successfully.") // 成功ログ
        
        // Draw image into context, adjusting for coordinate differences
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsPopContext()
        print("[toCVPixelBuffer]: Image drawn into context.") // 成功ログ
        
        return unwrappedPixelBuffer
    }
}

// CameraManager クラス (変更なし)
class CameraManager: NSObject, ObservableObject {
    @Published var currentFrame: UIImage?
    @Published var isRunning = false
    
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    
    override init() {
        super.init()
        setupCaptureSession()
    }
    
    private func setupCaptureSession() {
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("カメラにアクセスできません")
            return
        }
        
        captureSession.beginConfiguration()
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        captureSession.sessionPreset = .hd1280x720
        captureSession.commitConfiguration()
    }
    
    func startCapture() {
        guard !isRunning else { return }
        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
            DispatchQueue.main.async { self?.isRunning = true }
        }
    }
    
    func stopCapture() {
        guard isRunning else { return }
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
            DispatchQueue.main.async { self?.isRunning = false }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)
        guard let capturedCGImage = cgImage else { return }
        let image = UIImage(cgImage: capturedCGImage)
        DispatchQueue.main.async { [weak self] in self?.currentFrame = image }
    }
}

// CameraPreviewView (変更なし)
struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
}

// StyleThumbnailView (変更なし)
struct StyleThumbnailView: View {
    let styleName: String
    let isSelected: Bool
    
    var body: some View {
        VStack {
            ZStack {
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .aspectRatio(1, contentMode: .fit)
                    .cornerRadius(8)
                
                Text(styleName)
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(4)
            }
        }
        .padding(4)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 3)
        )
    }
}

struct ContentView: View {
    // @StateObject, @State 変数 (変更なし)
    @StateObject private var cameraManager = CameraManager()
    @State private var selectedDomain: String = domains.first!
    @State private var inputImage: UIImage? = nil
    @State private var convertedImage: UIImage? = nil
    @State private var isProcessing = false
    @State private var processingDuration: TimeInterval = 0
    @State private var debugMessages: [String] = []
    @State private var showDebugInfo = true
    
    // モデル (クラス名は要確認・修正)
    private let model: StarGANv2_256
    
    // 参照画像キャッシュ (変更なし)
    @State private var referenceImageCache: [String: UIImage] = [:]
    
    init() {
        print("ℹ️ ContentView initializing...")
        do {
            let configuration = MLModelConfiguration()
            model = try StarGANv2_256(configuration: configuration) // <<< モデルクラス名を確認・修正
            print("✅ Model loaded successfully.")
            print("Input Descriptions: \(model.model.modelDescription.inputDescriptionsByName)")
            print("Output Descriptions: \(model.model.modelDescription.outputDescriptionsByName)")
        } catch {
            print("❌ Fatal Error: Failed to load model: \(error.localizedDescription)")
            fatalError("Failed to load model: \(error.localizedDescription)")
        }
        print("ℹ️ ContentView initialized.")
    }
    
    // MARK: - Body
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                // 画像表示エリア
                HStack(spacing: 0) {
                    // --- 1a. 入力画像（左側）---
                    // 表示する画像が inputImage (前処理済み) に変わった
                    ZStack {
                        Color.gray.opacity(0.1).edgesIgnoringSafeArea(.top)
                        if let image = inputImage { // inputImage を表示
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else {
                            Rectangle().fill(Color.gray.opacity(0.3))
                        }
                        VStack { Spacer(); Text("入力 (モデル用)").font(.caption).foregroundColor(.white).padding(.horizontal, 6).padding(.vertical, 2).background(Color.black.opacity(0.6)).cornerRadius(4).padding(5) } // ラベル変更
                    }
                    .frame(width: geometry.size.width / 2)
                    .clipped()
                    
                    // --- 1b. 変換後の画像（右側）--- (変更なし)
                    ZStack {
                        Color.gray.opacity(0.1).edgesIgnoringSafeArea(.top)
                        if let image = convertedImage {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else {
                            Rectangle().fill(Color.gray.opacity(0.3))
                        }
                        VStack { Spacer(); Text("出力").font(.caption).foregroundColor(.white).padding(.horizontal, 6).padding(.vertical, 2).background(Color.black.opacity(0.6)).cornerRadius(4).padding(5) }
                    }
                    .frame(width: geometry.size.width / 2)
                    .clipped()
                }
                .frame(height: geometry.size.height * 0.7)
                
                // --- 2. スタイル選択部分 ---
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHGrid(rows: [GridItem(.fixed(80))], spacing: 10) { // 高さを固定
                        ForEach(domains, id: \.self) { domain in
                            StyleThumbnailView(styleName: domain, isSelected: selectedDomain == domain)
                                .onTapGesture {
                                    if selectedDomain != domain {
                                        selectedDomain = domain
                                        addDebugMessage("👉 Style changed to: \(domain)")
                                        // オプション: スタイル変更時に即時変換
                                        // if let currentFrame = cameraManager.currentFrame, !isProcessing {
                                        //     convertImage(image: currentFrame)
                                        // }
                                    }
                                }
                        }
                    }
                    .padding(.horizontal) // 左右に余白
                    .padding(.vertical, 5) // 上下に少し余白
                }
                .frame(height: 90) // ScrollView自体の高さを固定 (GridItemの高さ + padding)
                .background(Color.black.opacity(0.85)) // 背景色を少し濃く
                
                // --- 3. デバッグ情報表示 ---
                VStack(alignment: .leading, spacing: 4) { // デバッグ情報全体のVStack
                    // --- 3a. ヘッダーと開閉ボタン ---
                    HStack {
                        Text("デバッグ情報")
                            .font(.headline)
                            .foregroundColor(.white)
                        Spacer() // ボタンを右寄せ
                        Button {
                            withAnimation { // 開閉アニメーション
                                showDebugInfo.toggle()
                            }
                        } label: {
                            Image(systemName: showDebugInfo ? "chevron.up.circle.fill" : "chevron.down.circle.fill")
                                .foregroundColor(.gray)
                                .font(.title3) // アイコンサイズ調整
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top, 5) // 上部に少し余白
                    
                    // --- 3b. デバッグ内容 (表示状態に応じて表示) ---
                    if showDebugInfo {
                        ScrollView(.vertical) { // 縦スクロール可能にする
                            VStack(alignment: .leading, spacing: 4) {
                                // 処理時間
                                Text("処理時間: \(String(format: "%.3f", processingDuration))秒")
                                    .font(.caption)
                                    .foregroundColor(.gray) // 少し色を薄く
                                
                                Divider().background(Color.gray) // 区切り線
                                
                                // デバッグメッセージ (新しいものが上に来るように逆順表示)
                                ForEach(debugMessages.reversed().indices, id: \.self) { index in
                                    Text(debugMessages.reversed()[index])
                                        .font(.system(size: 10, design: .monospaced)) // 等幅フォント
                                        .foregroundColor(.white)
                                        .lineLimit(2) // 2行まで表示
                                        .frame(maxWidth: .infinity, alignment: .leading) // 左寄せ
                                }
                            }
                            .padding(.horizontal) // スクロール内の左右余白
                            .padding(.bottom, 5) // スクロール内の下部余白
                        }
                        // ScrollViewの最大高さを設定して、伸びすぎないようにする
                        .frame(maxHeight: 100)
                    }
                }
                .background(Color.black.opacity(0.85)) // デバッグエリア全体の背景
                
                // --- Spacerを追加してデバッグ情報を下部に押しやる (オプション) ---
                // Spacer()
                
            } // 全体のVStack
            .background(Color.black) // 背景色
            .edgesIgnoringSafeArea(.bottom) // 下のSafeAreaを無視
            .onAppear {
                addDebugMessage("ContentView appeared. Starting camera...")
                cameraManager.startCapture()
            }
            .onDisappear {
                addDebugMessage("ContentView disappeared. Stopping camera...")
                cameraManager.stopCapture()
            }
            .onChange(of: cameraManager.currentFrame) { newFrame in
                // メインスレッドで実行を保証 (より安全)
                DispatchQueue.main.async {
                    guard let frame = newFrame else { return }
                    if !isProcessing {
                        convertImage(image: frame)
                    }
                }
            }
        } // GeometryReader
    } // body
    
    // デバッグメッセージ追加 (メインスレッドで実行)
    private func addDebugMessage(_ message: String) {
        DispatchQueue.main.async {
            let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
            // 絵文字を追加して視認性向上
            let prefix: String
            if message.starts(with: "Error") || message.starts(with: "❌") { prefix = "❌ " }
            else if message.starts(with: "Warning") || message.starts(with: "⚠️") { prefix = "⚠️ " }
            else if message.starts(with: "✅") { prefix = "✅ " }
            else if message.starts(with: "👉") { prefix = "👉 " }
            else { prefix = "ℹ️ " } // 情報
            
            let fullMessage = "[\(timestamp)] \(prefix)\(message)"
            debugMessages.append(fullMessage)
            if debugMessages.count > 100 { // ログ件数制限
                debugMessages.removeFirst(debugMessages.count - 100)
            }
            print(fullMessage) // コンソールにも常に出力
        }
    }
    
    // 参照画像取得 (ログ追加)
    private func getReferenceImage(for domain: String) -> UIImage? {
        if let cachedImage = referenceImageCache[domain] {
            addDebugMessage("Reference image cache hit for: \(domain)")
            return cachedImage
        }
        addDebugMessage("Reference image cache miss for: \(domain). Loading from Assets...")
        guard let loadedImage = UIImage(named: domain) else {
            addDebugMessage("Error: Failed to load reference image from Assets for: \(domain)")
            return nil
        }
        addDebugMessage("Reference image loaded from Assets: \(domain)")
        DispatchQueue.main.async {
            referenceImageCache[domain] = loadedImage // キャッシュ更新
        }
        return loadedImage
    }
    
    // --- 追加: 全参照画像を非同期で事前にロードする関数（オプション） ---
    private func loadReferenceImagesAsync() {
        DispatchQueue.global(qos: .background).async {
            addDebugMessage("全参照画像の事前ロード開始...")
            var loadedCount = 0
            for domain in domains {
                if referenceImageCache[domain] == nil { // まだキャッシュされていない場合のみロード
                    if let loadedImage = UIImage(named: domain) {
                        // キャッシュに保存 (メインスレッドで)
                        DispatchQueue.main.async {
                            self.referenceImageCache[domain] = loadedImage
                            loadedCount += 1
                        }
                    } else {
                        addDebugMessage("Warning: 事前ロード失敗: \(domain)")
                    }
                }
            }
            addDebugMessage("全参照画像の事前ロード完了 (\(loadedCount)件)")
        }
    }
    
    // MARK: - Image Conversion Logic
    private func convertImage(image: UIImage) { // image はカメラからの生フレーム
        guard !isProcessing else { return }
        
        let startTime = Date()
        let currentSelectedDomain = selectedDomain
        addDebugMessage("Starting image conversion for style: \(currentSelectedDomain)")
        
        guard let domainIndex = domains.firstIndex(of: currentSelectedDomain) else {
            addDebugMessage("Error: Could not find domain index for: \(currentSelectedDomain)")
            return
        }
        let targetSize = CGSize(width: 256, height: 256)
        
        DispatchQueue.main.async { self.isProcessing = true; self.processingDuration = 0 }
        
        DispatchQueue.global(qos: .userInitiated).async {
            var finalConvertedImage: UIImage? = nil
            var finalProcessingTime: TimeInterval = 0
            var success = false
            
            do {
                // --- 修正: 入力画像の前処理 ---
                addDebugMessage("[1/7] Rotating and preprocessing (resizeAspectFill) input image...")
                guard let rotatedImage = image.rotateRight(),
                      // ★★★ resizeAspectFill を使用 ★★★
                      let preprocessedInputImage = rotatedImage.resizeAspectFill(to: targetSize) else {
                    throw ConversionError.preprocessingFailed("Input image rotation or resizeAspectFill failed.")
                }
                // ★★★ メインスレッドで前処理済み画像 inputImage を更新 ★★★
                DispatchQueue.main.async { self.inputImage = preprocessedInputImage }
                addDebugMessage("✅ [1/7] Input image preprocessed.")
                // --- 修正ここまで ---
                
                // --- 修正: CVPixelBufferには前処理済み画像を使う ---
                addDebugMessage("[2/7] Converting preprocessed input image to CVPixelBuffer...")
                guard let sourcePixelBuffer = preprocessedInputImage.toCVPixelBuffer(size: targetSize) else { // ★★★ preprocessedInputImage を使用 ★★★
                    throw ConversionError.bufferCreationFailed("Input image CVPixelBuffer creation failed.")
                }
                addDebugMessage("✅ [2/7] Input image CVPixelBuffer created.")
                // --- 修正ここまで ---
                
                
                addDebugMessage("[3/7] Getting reference image for \(currentSelectedDomain)...")
                guard let referenceUIImage = self.getReferenceImage(for: currentSelectedDomain) else {
                    throw ConversionError.referenceImageLoadFailed("Reference UIImage load failed for \(currentSelectedDomain).")
                }
                addDebugMessage("✅ [3/7] Reference UIImage loaded.")
                
                // --- 修正: 参照画像も resizeAspectFill を使う（任意）---
                // アスペクト比を維持したい場合はこちらも変更。そうでなければ resize のままでもOK。
                addDebugMessage("[4/7] Preprocessing (resizeAspectFill) reference image...")
                guard let preprocessedRefImage = referenceUIImage.resizeAspectFill(to: targetSize) else { // ★★★ resizeAspectFill を使用 ★★★
                    throw ConversionError.preprocessingFailed("Reference image resizeAspectFill failed.")
                }
                addDebugMessage("✅ [4/7] Reference image preprocessed.")
                // --- 修正ここまで ---
                
                
                // --- 修正: CVPixelBufferには前処理済み参照画像を使う ---
                addDebugMessage("[5/7] Converting preprocessed reference image to CVPixelBuffer...")
                guard let referencePixelBuffer = preprocessedRefImage.toCVPixelBuffer(size: targetSize) else { // ★★★ preprocessedRefImage を使用 ★★★
                    throw ConversionError.bufferCreationFailed("Reference image CVPixelBuffer creation failed.")
                }
                addDebugMessage("✅ [5/7] Reference image CVPixelBuffer created.")
                // --- 修正ここまで ---
                
                addDebugMessage("[6/7] Preparing domain index MLMultiArray...")
                guard let domainIndexArray = try? MLMultiArray(shape: [1] as [NSNumber], dataType: .int32) else {
                    throw ConversionError.multiArrayCreationFailed("Domain index MLMultiArray creation failed.")
                }
                domainIndexArray[0] = NSNumber(value: Int32(domainIndex))
                addDebugMessage("✅ [6/7] Domain index MLMultiArray prepared (index: \(domainIndex)).")
                
                
                addDebugMessage("[7/7] Performing model prediction...")
                // --- Inputクラス名 (要確認・修正) ---
                let input = StarGANv2_256Input(
                    source_image: sourcePixelBuffer,
                    reference_image: referencePixelBuffer,
                    reference_domain_index: domainIndexArray
                )
                
                let output = try self.model.prediction(input: input)
                addDebugMessage("✅ [7/7] Model prediction successful.")
                
                // --- 出力名 (要確認・修正) ---
                let outputMultiArray = output.generated_image_tensor
                addDebugMessage("Output tensor obtained, shape: \(outputMultiArray.shape). Converting to UIImage...")
                
                guard let outputUIImage = self.convertMultiArrayToUIImage(multiArray: outputMultiArray) else {
                    throw ConversionError.postprocessingFailed("Failed to convert output MLMultiArray to UIImage.")
                }
                addDebugMessage("✅ Output UIImage conversion successful.")
                
                // 成功時の結果を保持
                finalConvertedImage = outputUIImage
                finalProcessingTime = Date().timeIntervalSince(startTime)
                success = true
                
            } catch let error as ConversionError { // 自作エラー
                self.addDebugMessage("Error during conversion: \(error.localizedDescription)")
            } catch { // その他のエラー (CoreMLエラーなど)
                let nsError = error as NSError
                self.addDebugMessage("❌ Error during prediction: \(error.localizedDescription)")
                self.addDebugMessage("   Domain: \(nsError.domain), Code: \(nsError.code)")
                self.addDebugMessage("   UserInfo: \(nsError.userInfo)")
                print("❌ Prediction Error Details: \(error)") // コンソールに詳細
            }
            
            // --- UI更新 (メインスレッド) ---
            DispatchQueue.main.async {
                if success {
                    self.convertedImage = finalConvertedImage
                    self.processingDuration = finalProcessingTime
                    self.addDebugMessage("✅ Conversion finished successfully. Time: \(String(format: "%.3f", finalProcessingTime))s")
                } else {
                    // エラー発生時は出力画像をクリアする（オプション）
                    // self.convertedImage = nil
                    self.addDebugMessage("❌ Conversion failed.")
                }
                // 処理完了フラグを必ず下ろす
                self.isProcessing = false
            }
        }
    }
    
    
    // MLMultiArrayからUIImageへの変換
    private func convertMultiArrayToUIImage(multiArray: MLMultiArray) -> UIImage? {
        addDebugMessage("Converting MLMultiArray to UIImage (Input range: -1 to 1 assumed)")
        guard multiArray.dataType == .float32 else {
            addDebugMessage("Error [convertMultiArray]: MLMultiArray dataType is not Float32 (\(multiArray.dataType))")
            return nil
        }
        
        // --- 修正: オプショナルバインディングを削除し、直接代入とチェック ---
        guard multiArray.shape.count == 4, // 次元数チェック
              multiArray.shape[0].intValue == 1, // バッチサイズチェック
              multiArray.shape[1].intValue == 3  // チャンネル数チェック
        else {
            addDebugMessage("Error [convertMultiArray]: MLMultiArray shape dimension count or first dimensions are incorrect: \(multiArray.shape). Expected 4 dimensions starting with [1, 3, ...]")
            return nil
        }
        
        // shape.count == 4が保証されているので、インデックス2と3は存在する
        let height = multiArray.shape[2].intValue // 直接代入
        let width = multiArray.shape[3].intValue  // 直接代入
        
        // サイズが期待通りかチェック
        guard height == 256, width == 256 else {
            addDebugMessage("Error [convertMultiArray]: MLMultiArray height or width is incorrect: \(multiArray.shape). Expected [1, 3, 256, 256]")
            return nil
        }
        // --- 修正ここまで ---
        
        
        let count = width * height
        let channelStride = count // For CHW layout
        
        let rawPointer = multiArray.dataPointer
        let bufferPointer = UnsafeMutableBufferPointer<Float32>(start: rawPointer.assumingMemoryBound(to: Float32.self),
                                                                count: 3 * count) // Total number of floats
        
        var pixelData = [UInt8](repeating: 0, count: count * 4) // RGBA buffer
        
        // ... (以降のピクセル処理、CGImage作成は変更なし) ...
        for i in 0..<count {
            guard i < bufferPointer.count,
                  channelStride + i < bufferPointer.count,
                  channelStride * 2 + i < bufferPointer.count else {
                addDebugMessage("Error [convertMultiArray]: Index out of bounds while accessing buffer pointer at index \(i).")
                return nil
            }
            
            let rTanh = bufferPointer[i]
            let gTanh = bufferPointer[channelStride + i]
            let bTanh = bufferPointer[channelStride * 2 + i]
            
            let r_01 = (rTanh + 1.0) / 2.0
            let g_01 = (gTanh + 1.0) / 2.0
            let b_01 = (bTanh + 1.0) / 2.0
            
            let r = UInt8(clamping: Int(round(r_01 * 255.0)))
            let g = UInt8(clamping: Int(round(g_01 * 255.0)))
            let b = UInt8(clamping: Int(round(b_01 * 255.0)))
            
            let offset = i * 4
            pixelData[offset]     = r
            pixelData[offset + 1] = g
            pixelData[offset + 2] = b
            pixelData[offset + 3] = 255
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let bitsPerComponent = 8
        let bitsPerPixel = 32
        let bytesPerRow = width * 4
        
        guard let providerRef = CGDataProvider(data: Data(bytes: pixelData, count: count * 4) as CFData) else {
            addDebugMessage("Error [convertMultiArray]: Failed to create CGDataProvider.")
            return nil
        }
        
        guard let cgImage = CGImage(
            width: width, height: height, bitsPerComponent: bitsPerComponent, bitsPerPixel: bitsPerPixel, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo, provider: providerRef, decode: nil, shouldInterpolate: true, intent: .defaultIntent
        ) else {
            addDebugMessage("Error [convertMultiArray]: Failed to create CGImage.")
            return nil
        }
        addDebugMessage("✅ UIImage conversion successful.")
        return UIImage(cgImage: cgImage)
    }
}

// --- 追加: エラー型定義 ---
enum ConversionError: Error, LocalizedError {
    case preprocessingFailed(String)
    case bufferCreationFailed(String)
    case referenceImageLoadFailed(String)
    case multiArrayCreationFailed(String)
    case postprocessingFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .preprocessingFailed(let reason): return "Preprocessing failed: \(reason)"
        case .bufferCreationFailed(let reason): return "CVPixelBuffer creation failed: \(reason)"
        case .referenceImageLoadFailed(let reason): return "Reference image load failed: \(reason)"
        case .multiArrayCreationFailed(let reason): return "MLMultiArray creation failed: \(reason)"
        case .postprocessingFailed(let reason): return "Postprocessing failed: \(reason)"
        }
    }
}
