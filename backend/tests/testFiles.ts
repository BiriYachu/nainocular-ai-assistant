// Helper for tests: builds a tiny valid PDF in memory.
// Each item in pageTexts is the text of one page. Use null for a page with no text at all.

export function buildPdf(pageTexts: (string | null)[]): Buffer {
  const objects: string[] = [];
  const pageCount = pageTexts.length;
  const kids = pageTexts.map((_, i) => `${4 + i * 2} 0 R`).join(" ");

  objects[1] = "<< /Type /Catalog /Pages 2 0 R >>";
  objects[2] = `<< /Type /Pages /Kids [${kids}] /Count ${pageCount} >>`;
  objects[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";

  pageTexts.forEach((text, i) => {
    const pageNumber = 4 + i * 2;
    const contentNumber = pageNumber + 1;
    if (text === null) {
      objects[pageNumber] = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"; // blank page
      objects[contentNumber] = "<< >>";
    } else {
      objects[pageNumber] =
        `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents ${contentNumber} 0 R >>`;
      const stream = `BT /F1 12 Tf 50 700 Td (${text}) Tj ET`;
      objects[contentNumber] = `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`;
    }
  });

  let pdf = "%PDF-1.4\n";
  const offsets: number[] = [];
  for (let i = 1; i < objects.length; i++) {
    offsets[i] = pdf.length;
    pdf += `${i} 0 obj\n${objects[i]}\nendobj\n`;
  }
  const xrefStart = pdf.length;
  pdf += `xref\n0 ${objects.length}\n0000000000 65535 f \n`;
  for (let i = 1; i < objects.length; i++) {
    pdf += String(offsets[i]).padStart(10, "0") + " 00000 n \n";
  }
  pdf += `trailer\n<< /Size ${objects.length} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`;
  return Buffer.from(pdf, "latin1");
}
