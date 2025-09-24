import sys
from rdkit import Chem
from rdkit.Chem import Draw

def generate_image(smiles, output_path):
    try:
        # 解析SMILES字符串
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error: Invalid SMILES string: {smiles}")
            sys.exit(1)

        # 生成分子图像
        img = Draw.MolsToGridImage(
            [mol],
            molsPerRow=1,
            subImgSize=(300, 300),
            legends=[""]  # 将legends设为空字符串，这样就不会显示标注
        )

        # 保存图像
        img.save(output_path)
        print(f"Image generated successfully: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_molecule_image.py <smiles> <output_path>")
        sys.exit(1)

    smiles = sys.argv[1]
    output_path = sys.argv[2]

    generate_image(smiles, output_path)