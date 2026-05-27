@echo off
echo ====================================================
echo KICH HOAT PIPELINE RAG TREN WINDOWS
echo ====================================================

call rag_env\Scripts\activate.bat

echo [1/3] Baseline (BM25 only)...
python src\main.py --mode baseline0 --input data\test\questions.txt --output system_outputs\system_output_3.txt

echo [2/3] Dense RAG (FAISS only)...
python src\main.py --mode dense_only --input data\test\questions.txt --output system_outputs\system_output_2.txt

echo [3/3] Advanced Hybrid RAG (BM25 + FAISS)...
python src\main.py --mode advanced_full --input data\test\questions.txt --output system_outputs\system_output_1.txt

echo.
echo ========== DANH GIA KET QUA ==========
echo.
echo --- Advanced Hybrid RAG ---
python src\evaluate.py --pred system_outputs\system_output_1.txt --gold data\test\reference_answers.txt
echo.
echo --- Dense RAG (FAISS only) ---
python src\evaluate.py --pred system_outputs\system_output_2.txt --gold data\test\reference_answers.txt
echo.
echo --- Baseline (BM25 only) ---
python src\evaluate.py --pred system_outputs\system_output_3.txt --gold data\test\reference_answers.txt
echo.
echo ====================================================
echo HOAN TAT! KET QUA LUU TRONG system_outputs\
echo ====================================================
pause