#!/bin/bash
#SBATCH --job-name=download_flux2
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/download_flux2_%j.out
#SBATCH --error=jobs/logs/download_flux2_%j.err

echo "======================================================================"
echo "DOWNLOAD FLUX.2-KLEIN-base-4B MODEL"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "======================================================================"

# Activer l'environnement virtuel
source ~/text-in-image-generation/venv/bin/activate

# Aller à la racine du projet
cd ~/text-in-image-generation

# Configuration Hugging Face
export HF_HOME=$HOME/text-in-image-generation/.hf_cache
# HF_HUB_ENABLE_HF_TRANSFER désactivé (nécessite hf_transfer package)

#export HF_TOKEN="token"

# Aller dans le dossier de la tâche
cd ~/text-in-image-generation/Finetune_with_LoRA

# Lancer le téléchargement
python download_model.py

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Fin du job: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: ✓ MODÈLE TÉLÉCHARGÉ AVEC SUCCÈS"
    echo ""
    echo "Vérification finale:"
    python check_model.py
    echo ""
    echo "Taille du modèle:"
    du -sh models/flux2-klein-base-4b/
else
    echo "Status: ✗ ERREUR LORS DU TÉLÉCHARGEMENT"
    echo "Vérifiez les logs ci-dessus"
    echo ""
    echo "Causes possibles:"
    echo "  1. Modèle gated - token requis (décommentez HF_TOKEN)"
    echo "  2. Problème réseau - relancez le job (reprendra automatiquement)"
    echo "  3. Espace disque insuffisant - vérifiez: df -h ~"
fi

echo "======================================================================"

exit $EXIT_CODE