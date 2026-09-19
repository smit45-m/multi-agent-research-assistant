# Biomedical & Genomics: Technical Reference

## CRISPR-Cas9 Base Editing

Base editors fuse a catalytically impaired Cas9 nickase to a deaminase
enzyme — cytidine deaminase (CBE) converts C-to-T, adenine deaminase (ABE)
converts A-to-G — enabling correction of a single nucleotide variant without
double-strand breaks. The guide RNA positions the editor; targeting requires
a suitable pam sequence (NGG for SpCas9) placing the target base in the
editing window. Fidelity depends on off-target deamination: engineered
variants (ABE8e, YE1-CBE) narrow the editing window and cut Cas-independent
off-target cleavage and RNA editing. Clinical programs (sickle cell disease,
PCSK9 hypercholesterolemia) report potent in vivo editing with reduced
off-target profiles versus nuclease editing.

## Lipid Nanoparticle (LNP) mRNA Delivery

LNPs package mRNA in four lipids: an ionizable lipid that is neutral in
circulation but protonates in acidic endosomes to drive endosomal escape; a
pegylated lipid controlling particle size and circulation time; cholesterol;
and a helper phospholipid. Encapsulation efficiency above 90 percent is
standard with microfluidic mixing. Tissue tropism defaults to liver via ApoE
opsonization; selective organ targeting lipids redirect to lung or spleen.
Reactogenicity and anti-PEG immunogenicity are managed through lipid
clearance engineering; the ionizable lipid pKa (6.2-6.8) is the strongest
determinant of potency and tolerability.

## AlphaFold Protein Structure Prediction

AlphaFold2 predicts tertiary structure from a multiple sequence alignment:
the evoformer trunk exchanges information between MSA and pairwise residue
representations, and the structure module places residues with invariant
point attention, an SE(3)-equivariant mechanism. Confidence is expressed
per-residue as the pLDDT score (predicted lDDT, 0-100) and pairwise as PAE.
Limitations: static single conformations, weak conformational dynamics,
disordered regions score low pLDDT, and small-molecule docking against
predicted structures degrades when side-chain placement is imperfect —
motivating diffusion models that jointly predict complexes with ligands.

## CAR-T Antigen Escape

CAR-T therapy in hematologic malignancies fails through antigen escape:
antigen loss variants (CD19 splice isoforms lacking the target epitope),
cd19 modulation via lineage switch or trogocytosis, and low-density antigen
below CAR activation thresholds. T-cell exhaustion from tonic CAR signaling
erodes persistence; 4-1BB costimulation favors memory phenotypes while CD28
gives faster but shorter effector responses. Cytokine release syndrome and
neurotoxicity are managed with tocilizumab and steroids. Countermeasures
include bispecific car designs (CD19/CD22 dual targeting), tandem CARs, and
armored CARs secreting IL-15.

## Circular mRNA Therapeutics

mRNA circularization removes free ends recognized by exonucleases, giving
half-life extension from hours to days. The dominant method uses a group i
intron permuted intron-exon (PIE) system for autocatalytic splicing that
joins the ends into circular rna; cap-independent translation proceeds from
an internal ribosome entry site (IRES). Benefits: translation persistence
severalfold longer than linear mRNA and lower innate immunogenicity when
purified of linear byproducts (RNase R digestion, HPLC). Challenges are
circularization efficiency at scale, IRES strength tuning per cell type,
and removing double-stranded RNA contaminants that trigger RIG-I.

## Single-Cell RNA Sequencing Analysis

Droplet based platforms (10x Chromium) encapsulate single cells with
barcoded beads, resolving cellular heterogeneity invisible to bulk RNA-seq.
Analysis pipelines normalize counts, select variable genes, and embed cells
with umap embedding for visualization; clustering (Leiden) identifies cell
types. Trajectory inference orders cells along developmental paths:
pseudotime methods such as monocle fit principal graphs, while RNA velocity
uses spliced/unspliced ratios to infer direction. Pitfalls include ambient
RNA contamination, doublets, batch effects (corrected with Harmony/scVI),
and over-interpretation of UMAP distances.

## Liquid Biopsy ctDNA Detection

Liquid biopsy detects circulating tumor DNA fragments shed into plasma.
next generation sequencing panels with unique molecular identifiers and
error suppression reach a sensitivity threshold of 0.01-0.1 percent variant
allele fraction; digital pcr offers absolute quantification for known
hotspot somatic mutation targets at similar sensitivity with faster
turnaround. Applications: early detection (multi-cancer screening using
methylation signatures), minimal residual disease after surgery, and
resistance monitoring (EGFR T790M). Specificity requires filtering clonal
hematopoiesis variants via matched white-cell sequencing.

## Antibody-Drug Conjugates (ADCs)

ADCs couple a targeting antibody to a cytotoxic payload through a linker.
A cleavable linker (valine-citrulline dipeptide, hydrazone, or disulfide)
releases payload inside the cell and enables bystander killing of
neighboring tumor cells; non-cleavable linkers demand full lysosomal
degradation but improve plasma stability. The dar ratio (drug-to-antibody
ratio, typically 2-8) balances potency against aggregation and clearance.
Target selectivity hinges on antigen overexpression on tumor versus normal
tissue; pharmacokinetics are dominated by the antibody but off-target
payload release drives dose-limiting toxicities such as neutropenia and
ocular effects.

## Microbiome Short-Chain Fatty Acids

Gut bacteria ferment dietary fiber into short-chain fatty acids — acetate,
propionate, and butyrate. Butyrate is the preferred fuel of colonocytes and
reinforces the epithelial barrier by upregulating tight junctions proteins
(claudins, occludin) and mucin production. SCFAs signal through the gpr43
receptor (FFAR2) and GPR109A on immune cells, expanding regulatory T cells
and providing inflammation modulation; they also inhibit histone
deacetylases. Systemically, SCFAs influence the gut-brain axis via vagal
afferents, microglial maturation, and blood-brain-barrier integrity,
linking fiber intake to metabolic and neuro-immune outcomes.

## Tau Phosphorylation in Neurodegeneration

Tau stabilizes axonal microtubules; kinase-phosphatase imbalance (GSK-3beta,
CDK5 versus PP2A) causes hyperphosphorylation, detaching tau and collapsing
microtubule stability. Detached tau misfolds into oligomers — oligomer
toxicity now appears greater than that of mature neurofibrillary tangles —
and propagates trans-synaptically in prion-like fashion. Clearance fails as
the glymphatic system (perivascular CSF-ISF exchange, most active in sleep)
declines with age; reactive astrogliosis and microglial activation
amplify injury. Therapeutics target phospho-tau epitopes (immunotherapy),
aggregation inhibitors, and antisense reduction of total tau.
