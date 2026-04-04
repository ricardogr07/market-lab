document.addEventListener("DOMContentLoaded", () => {
  const mermaidBlocks = Array.from(
    document.querySelectorAll("pre code.language-mermaid"),
  );

  if (mermaidBlocks.length === 0 || typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
  });

  mermaidBlocks.forEach((block, index) => {
    const pre = block.parentElement;
    if (!pre || !pre.parentElement) {
      return;
    }

    const container = document.createElement("div");
    container.className = "mermaid";
    container.id = `mermaid-diagram-${index}`;
    container.textContent = block.textContent ?? "";
    pre.replaceWith(container);
  });

  mermaid.run({
    querySelector: ".mermaid",
  });
});
