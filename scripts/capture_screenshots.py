import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By


def capture_all():
    screenshots_dir = Path("reports/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    options = EdgeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Edge(options=options)

    try:
        driver.get("http://localhost:8501")
        # wait for streamlit reactive dom to render
        time.sleep(6)

        driver.save_screenshot(str(screenshots_dir / "04_streamlit_prediction.png"))

        try:
            el2 = driver.find_element(By.XPATH, "//*[contains(text(), 'Benchmarks')]")
            driver.execute_script("arguments[0].click();", el2)
            time.sleep(3)

            driver.save_screenshot(str(screenshots_dir / "01_metrics_dashboard.png"))
            driver.save_screenshot(str(screenshots_dir / "05_parameters_and_evaluation.png"))

            driver.execute_script("window.scrollBy(0, 450);")
            time.sleep(1)
            driver.save_screenshot(str(screenshots_dir / "02_confusion_matrix.png"))
        except Exception:
            pass

        try:
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            el3 = driver.find_element(By.XPATH, "//*[contains(text(), 'Ablation')]")
            driver.execute_script("arguments[0].click();", el3)
            time.sleep(3)

            driver.save_screenshot(str(screenshots_dir / "03_ablation_results.png"))
        except Exception:
            pass

    finally:
        driver.quit()


if __name__ == "__main__":
    capture_all()
