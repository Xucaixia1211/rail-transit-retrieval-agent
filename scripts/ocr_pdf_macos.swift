#!/usr/bin/env swift

import AppKit
import Foundation
import PDFKit
import Vision

struct OCRPage: Codable {
    let page: Int
    let text: String
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: ocr_pdf_macos.swift INPUT.pdf")
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard let document = PDFDocument(url: inputURL) else {
    fail("cannot open PDF: \(inputURL.path)")
}

let encoder = JSONEncoder()
for pageIndex in 0..<document.pageCount {
    guard let page = document.page(at: pageIndex) else { continue }
    let bounds = page.bounds(for: .mediaBox)
    let longestSide: CGFloat = 2600
    let scale = min(longestSide / max(bounds.width, bounds.height), 4.0)
    let imageSize = NSSize(
        width: max(bounds.width * scale, 1),
        height: max(bounds.height * scale, 1)
    )
    let image = page.thumbnail(of: imageSize, for: .mediaBox)
    var proposedRect = NSRect(origin: .zero, size: image.size)
    guard let cgImage = image.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil) else {
        fail("cannot render page \(pageIndex + 1)")
    }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["zh-Hans", "en-US"]
    if #available(macOS 13.0, *) {
        request.automaticallyDetectsLanguage = true
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
    } catch {
        fail("OCR failed on page \(pageIndex + 1): \(error)")
    }

    let observations = (request.results ?? []).sorted { left, right in
        let verticalDifference = abs(left.boundingBox.midY - right.boundingBox.midY)
        if verticalDifference > 0.012 {
            return left.boundingBox.midY > right.boundingBox.midY
        }
        return left.boundingBox.minX < right.boundingBox.minX
    }
    let lines = observations.compactMap { $0.topCandidates(1).first?.string }
    let result = OCRPage(page: pageIndex + 1, text: lines.joined(separator: "\n"))
    if let data = try? encoder.encode(result), let line = String(data: data, encoding: .utf8) {
        print(line)
        fflush(stdout)
    }
}
