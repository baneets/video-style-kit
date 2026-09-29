// ocr: Apple Vision text recognition on PNG/JPG files, on-device. usage: ocr <img>... -> one JSON line per image
// {"file":..., "lines":[{"text":..., "conf":0.9, "box":[x,y,w,h] (fractions, top-left origin), "words":[{"text","box"}]}]}
import Foundation
import Vision
import AppKit

func run(_ path: String) -> [String: Any] {
    guard let img = NSImage(contentsOfFile: path), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return ["file": path, "error": "unreadable"]
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    let h = VNImageRequestHandler(cgImage: cg, options: [:])
    try? h.perform([req])
    var lines: [[String: Any]] = []
    for r in req.results ?? [] {
        guard let c = r.topCandidates(1).first else { continue }
        let b = r.boundingBox  // normalized, bottom-left origin
        var words: [[String: Any]] = []
        let s = c.string
        var idx = s.startIndex
        for w in s.split(separator: " ") {
            if let range = s.range(of: String(w), range: idx..<s.endIndex), let wb = try? c.boundingBox(for: range) {
                let bb = wb.boundingBox
                words.append(["text": String(w), "box": [bb.minX, 1 - bb.maxY, bb.width, bb.height]])
                idx = range.upperBound
            }
        }
        lines.append(["text": s, "conf": c.confidence, "box": [b.minX, 1 - b.maxY, b.width, b.height], "words": words])
    }
    return ["file": path, "lines": lines]
}

func faces(_ path: String) -> [String: Any] {
    guard let img = NSImage(contentsOfFile: path), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return ["file": path, "error": "unreadable"]
    }
    let req = VNDetectFaceRectanglesRequest()
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    let out: [[String: Any]] = (req.results ?? []).map { r in
        let b = r.boundingBox
        return ["box": [b.minX, 1 - b.maxY, b.width, b.height], "conf": r.confidence]
    }
    return ["file": path, "faces": out]
}

var args = Array(CommandLine.arguments.dropFirst())
let mode = args.first == "faces" ? "faces" : "text"
if mode == "faces" { args.removeFirst() }
for p in args {
    let d = try! JSONSerialization.data(withJSONObject: mode == "faces" ? faces(p) : run(p))
    print(String(data: d, encoding: .utf8)!)
}
