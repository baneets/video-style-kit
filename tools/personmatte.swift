// personmatte: cut the person out of a video on-device with Apple Vision, no cloud, no credits.
// usage: personmatte <in.mov> <out.mov>   -> ProRes 4444 with alpha, same size/fps.
import AVFoundation
import Vision
import CoreImage
import CoreVideo

let args = CommandLine.arguments
guard args.count == 3 else { fputs("usage: personmatte in out\n", stderr); exit(2) }
let inURL = URL(fileURLWithPath: args[1]), outURL = URL(fileURLWithPath: args[2])
try? FileManager.default.removeItem(at: outURL)

let asset = AVURLAsset(url: inURL)
let track = asset.tracks(withMediaType: .video).first!
let size = track.naturalSize
let W = Int(size.width), H = Int(size.height)

let reader = try! AVAssetReader(asset: asset)
let rout = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
reader.add(rout); reader.startReading()

let writer = try! AVAssetWriter(outputURL: outURL, fileType: .mov)
let win = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.proRes4444, AVVideoWidthKey: W, AVVideoHeightKey: H])
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: win, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: W, kCVPixelBufferHeightKey as String: H])
writer.add(win); writer.startWriting(); writer.startSession(atSourceTime: .zero)

let ctx = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
let req = VNGeneratePersonSegmentationRequest()
req.qualityLevel = ProcessInfo.processInfo.environment["PERSONMATTE_QUALITY"] == "balanced" ? .balanced : .accurate
req.outputPixelFormat = kCVPixelFormatType_OneComponent8
var n = 0
while let sb = rout.copyNextSampleBuffer() {
    guard let px = CMSampleBufferGetImageBuffer(sb) else { continue }
    let pts = CMSampleBufferGetPresentationTimeStamp(sb)
    let handler = VNImageRequestHandler(cvPixelBuffer: px, options: [:])
    try! handler.perform([req])
    guard let mask = req.results?.first?.pixelBuffer else { continue }
    let src = CIImage(cvPixelBuffer: px)
    var m = CIImage(cvPixelBuffer: mask)
    m = m.transformed(by: CGAffineTransform(scaleX: src.extent.width / m.extent.width, y: src.extent.height / m.extent.height))
    // soften the edge a touch so hair does not look cut with scissors
    m = m.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 1.5]).cropped(to: src.extent)
    let out = src.applyingFilter("CIBlendWithMask", parameters: [
        kCIInputBackgroundImageKey: CIImage(color: .clear).cropped(to: src.extent), kCIInputMaskImageKey: m])
    var dst: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &dst)
    ctx.render(out, to: dst!)
    while !win.isReadyForMoreMediaData { usleep(2000) }
    adaptor.append(dst!, withPresentationTime: pts)
    n += 1
    if n % 100 == 0 { fputs("\(n) frames\n", stderr) }
}
win.markAsFinished()
let done = DispatchSemaphore(value: 0); writer.finishWriting { done.signal() }; done.wait()
print("wrote \(n) frames -> \(outURL.path)")
