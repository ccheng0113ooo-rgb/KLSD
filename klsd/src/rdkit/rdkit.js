// src/rdkit/rdkit.js
export async function getRdkitModule() {
  if (window.RDKit) {
    return window.RDKit;
  }
  
  return new Promise((resolve) => {
    const interval = setInterval(() => {
      if (window.RDKit) {
        clearInterval(interval);
        resolve(window.RDKit);
      }
    }, 100);
  });
}