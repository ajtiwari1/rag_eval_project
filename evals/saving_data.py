import json
from pathlib import Path
from datetime import datetime

def save_results(result, output_dir="evaluation_results", filename_prefix="retriever"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"{filename_prefix}_eval_{timestamp}.json"

    test_results = []

    for item in getattr(result, "test_results", []):
        metrics = []

        for metric in getattr(item, "metrics_data", []):
            metrics.append({
                "metric": getattr(metric, "name", None),
                "score": getattr(metric, "score", None),
                "threshold": getattr(metric, "threshold", None),
                "success": getattr(metric, "success", None),
                "reason": getattr(metric, "reason", None),
            })

        test_results.append({
            "input": getattr(item, "input", None),
            "success": getattr(item, "success", None),
            "metrics": metrics,
        })

    report = {
        "timestamp": datetime.now().isoformat(),
        "test_case_count": len(test_results),
        "test_results": test_results,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nEvaluation results saved to: {file_path}")
    return file_path