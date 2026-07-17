import os
import pandas as pd
import argparse

def generate_cub_stats(csv_path):
    # Automatically save the text file in the same directory as the CSV
    base_dir = os.path.dirname(csv_path)
    output_file = os.path.join(base_dir, "data-stats.txt")
    
    print(f"[*] Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Error loading CSV: {e}")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=====================================================================\n")
        f.write("             CUB-200 FINE-GRAINED VQA DATASET STATISTICS\n")
        f.write("=====================================================================\n\n")
        
        # ---------------------------------------------------------
        # 1. High-Level Metrics
        # ---------------------------------------------------------
        total_qa = len(df)
        unique_images = df['filename'].nunique()
        
        f.write("[1] HIGH-LEVEL METRICS\n")
        f.write("-" * 65 + "\n")
        f.write(f"Total QA Pairs                  : {total_qa:,}\n")
        f.write(f"Total Unique Images             : {unique_images:,}\n")
        if unique_images > 0:
            f.write(f"Average Questions per Image     : {total_qa / unique_images:.2f}\n")
            q_per_img = df.groupby('filename').size()
            f.write(f"Max Questions on a Single Image : {q_per_img.max()}\n")
            f.write(f"Min Questions on a Single Image : {q_per_img.min()}\n")
        f.write("\n")
        
        # ---------------------------------------------------------
        # 2. Extract Species/Classes from Filename
        # ---------------------------------------------------------
        # The filename format is: images/001.Black_footed_Albatross/...
        # Splitting by '/' and taking the second element gives us the species class.
        df['species_folder'] = df['filename'].apply(
            lambda x: x.split('/')[1] if isinstance(x, str) and '/' in x else 'Unknown'
        )
        unique_species = df['species_folder'].nunique()
        
        f.write("[2] BIRD SPECIES / CLASS METRICS\n")
        f.write("-" * 65 + "\n")
        f.write(f"Total Unique Bird Species       : {unique_species:,}\n")
        if unique_species > 0:
            f.write(f"Average Images per Species      : {unique_images / unique_species:.2f}\n")
            f.write(f"Average Questions per Species   : {total_qa / unique_species:.2f}\n")
        f.write("\n")
        
        # ---------------------------------------------------------
        # 3. Question Structural Analysis
        # ---------------------------------------------------------
        f.write("[3] QUESTION STRUCTURAL ANALYSIS\n")
        f.write("-" * 65 + "\n")
        
        # Distinguish multiple choice from open-ended
        mc_mask = df['question'].str.contains("Options:", na=False)
        mc_count = mc_mask.sum()
        open_count = total_qa - mc_count
        
        f.write(f"Multiple-Choice Questions       : {mc_count:,} ({mc_count/total_qa*100:.1f}%)\n")
        f.write(f"Open-Ended Questions            : {open_count:,} ({open_count/total_qa*100:.1f}%)\n")
        
        # Check for binary Yes/No tasks
        yes_no_count = df['answer'].astype(str).str.lower().isin(['yes', 'no']).sum()
        f.write(f"Binary (Yes/No) Answers         : {yes_no_count:,} ({yes_no_count/total_qa*100:.1f}%)\n")
        f.write("\n")
        
        # ---------------------------------------------------------
        # 4. Category Breakdown
        # ---------------------------------------------------------
        f.write("[4] DETAILED CATEGORY BREAKDOWN\n")
        f.write("-" * 65 + "\n")
        
        cat_counts = df['category'].value_counts()
        for cat, count in cat_counts.items():
            cat_df = df[df['category'] == cat]
            uniq_q = cat_df['question'].nunique()
            uniq_a = cat_df['answer'].nunique()
            cat_mc = cat_df['question'].str.contains("Options:", na=False).sum()
            
            f.write(f"➤ {cat}:\n")
            f.write(f"   • Total Questions : {count:,} ({count/total_qa*100:.1f}%)\n")
            f.write(f"   • Unique Questions: {uniq_q:,}\n")
            f.write(f"   • Unique Answers  : {uniq_a:,}\n")
            f.write(f"   • Multiple-Choice : {cat_mc:,}\n")
            f.write(f"   • Open-Ended      : {count - cat_mc:,}\n\n")
        
        f.write("=====================================================================\n")

    print(f"[+] Successfully generated detailed CUB-200 VQA stats at: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate detailed stats for the CUB-200 QA dataset.")
    parser.add_argument(
        "--csv_path", 
        type=str, 
        required=True,
        help="Absolute path to the QA.csv file"
    )
    args = parser.parse_args()
    
    generate_cub_stats(args.csv_path)