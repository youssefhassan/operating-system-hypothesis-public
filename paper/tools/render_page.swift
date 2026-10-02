// Render PDF pages to PNG with macOS PDFKit (no poppler needed): swift tools/render_page.swift paper.pdf 3,7,12 outprefix  -> outprefix_p3.png ...
import PDFKit
import AppKit
let args = CommandLine.arguments
let doc = PDFDocument(url: URL(fileURLWithPath: args[1]))!
let pages = args[2].split(separator: ",").map { Int($0)! - 1 }
for n in pages {
let page = doc.page(at: n)!
let r = page.bounds(for: .mediaBox)
let scale: CGFloat = 1.6
let img = NSImage(size: NSSize(width: r.width*scale, height: r.height*scale))
img.lockFocus()
NSColor.white.set(); NSRect(x:0,y:0,width:r.width*scale,height:r.height*scale).fill()
let ctx = NSGraphicsContext.current!.cgContext
ctx.scaleBy(x: scale, y: scale)
page.draw(with: .mediaBox, to: ctx)
img.unlockFocus()
let tiff = img.tiffRepresentation!
let rep = NSBitmapImageRep(data: tiff)!
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: args[3] + "_p\(n+1).png"))
}
print("ok", doc.pageCount)
