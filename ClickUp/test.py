from selenium import webdriver
from selenium.webdriver.common.by import By
import time, sys
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium_stealth import stealth
from pathlib import Path
tasks = {
  "Mathematics - Proof & Problem Solving": {
    "due_date": "2026-07-14T09:00:00",
    "duration_minutes": 45,
    "priority": "High",
    "subject": "Mathematics",
    "description": "Complete one mathematical proof question and two challenging extension problems. Show every logical step, justify every conclusion, and explain why each method works rather than only writing the answer. Spend the final 10 minutes reviewing your work and rewriting any solutions where reasoning was skipped."
  },

  "Geography - Extended Responses": {
    "due_date": "2026-07-14T09:45:00",
    "duration_minutes": 40,
    "priority": "Medium",
    "subject": "Geography",
    "description": "Practise climate graphs, anticyclones or the current class topic. Complete two extended-response questions using the structure: Point → Evidence/Example → Explanation → Consequence → Link back to the question. Focus on writing deeper explanations instead of short factual answers."
  },

  "Biology - Data Analysis": {
    "due_date": "2026-07-14T10:25:00",
    "duration_minutes": 35,
    "priority": "High",
    "subject": "Biology",
    "description": "Complete graph and data-analysis questions. Every answer should follow the structure: identify the trend, quote numerical evidence, then provide a biological explanation. Mark your answers afterwards and rewrite any responses that lost marks."
  },

  "History - Source Evaluation": {
    "due_date": "2026-07-14T11:00:00",
    "duration_minutes": 30,
    "priority": "Medium",
    "subject": "History",
    "description": "Complete one GCSE-style source usefulness question. Analyse the source by discussing its content, provenance, purpose, audience, reliability, limitations and overall usefulness. Support your judgement using contextual historical knowledge."
  },

  "French - Speaking & Listening": {
    "due_date": "2026-07-14T11:30:00",
    "duration_minutes": 30,
    "priority": "Medium",
    "subject": "French",
    "description": "Spend 10 minutes listening to native French audio, 10 minutes shadowing the speaker to improve pronunciation, and 10 minutes speaking without a script. Aim to answer questions using complete sentences and improve fluency rather than translating directly from English."
  }
}

for i in tasks.keys():
    automation_profile = Path.home() / "selenium-chrome-profile"
    automation_profile.mkdir(exist_ok=True)

    options = webdriver.ChromeOptions()
    options.add_argument(f"--user-data-dir={automation_profile}")
    options.add_argument("--profile-directory=Default")
    driver_1 = webdriver.Chrome(options=options)
    action_1 = ActionChains(driver_1)

    driver_1.get('https://app.clickup.com/90121884882/my-work/tasks')

    time.sleep(3)
    add_task = driver_1.find_element(By.XPATH, '//*[@id="app-root"]/cu-app-view/cu-app-shell/cu-manager/div/div/div/main/div/div/div/div/cu-my-work-core/cu-canvas-content/cu-canvas/gridstack/gridstack-item/div/cu-canvas-card/cu-plugin/cu-my-work-plugin/cu-assigned-to-me/cu-card/div/cu-list-view-instance/cu-dashboard-table/div/div/div/cu-list-view-divisions/cu-drop-area-to-create/div/div/cu-task-list/cu-task-list-footer/div[1]/cu-create-task-menu/div/button')
    add_task.click()
    time.sleep(0.5)
    task_name_entry = driver_1.find_element(By.XPATH, '//*[@id="app-root"]/cu-app-view/cu-app-shell/cu-manager/div/div/div/main/div/div/div/div/cu-my-work-core/cu-canvas-content/cu-canvas/gridstack/gridstack-item/div/cu-canvas-card/cu-plugin/cu-my-work-plugin/cu-assigned-to-me/cu-card/div/cu-list-view-instance/cu-dashboard-table/div/div/div/cu-list-view-divisions/cu-drop-area-to-create/div/div/cu-task-list/cu-task-row-new/div/div/div[1]/div/cu-slash-command/input')
    task_name_entry.send_keys(i)
    time.sleep(0.2)
    select_list_entry = driver_1.find_element(By.XPATH, '//*[@id="app-root"]/cu-app-view/cu-app-shell/cu-manager/div/div/div/main/div/div/div/div/cu-my-work-core/cu-canvas-content/cu-canvas/gridstack/gridstack-item/div/cu-canvas-card/cu-plugin/cu-my-work-plugin/cu-assigned-to-me/cu-card/div/cu-list-view-instance/cu-dashboard-table/div/div/div/cu-list-view-divisions/cu-drop-area-to-create/div/div/cu-task-list/cu-task-row-new/div/div/div[1]/div/cu-category-list-dropdown/cu-hierarchy-picker-dropdown/div/div/div')
    select_list_entry.click()
    time.sleep(0.1)
    list_search_entry = driver_1.find_element(By.XPATH, '//*[@id="cdk-overlay-0"]/div/cu-hierarchy-picker/div/div[1]/input')
    list_search_entry.send_keys('TasksBot')
    time.sleep(1.5)
    list_search_entry.send_keys(Keys.RETURN)


    time.sleep(1.5)
    print('running the rest')
    set_due_date = driver_1.find_element(By.XPATH, '//*[@id="app-root"]/cu-app-view/cu-app-shell/cu-manager/div/div/div/main/div/div/div/div/cu-my-work-core/cu-canvas-content/cu-canvas/gridstack/gridstack-item/div/cu-canvas-card/cu-plugin/cu-my-work-plugin/cu-assigned-to-me/cu-card/div/cu-list-view-instance/cu-dashboard-table/div/div/div/cu-list-view-divisions/cu-drop-area-to-create/div/div/cu-task-list/cu-task-row-new/div/div/div[2]/cu-task-row-recurring-date-picker/cu-recurring-date-dropdown/div/div/div/button')
    set_due_date.click()
    print('clicked set_due_date')
    time.sleep(0.5)
    tomorrow = driver_1.find_element(By. XPATH, '//*[@id="cdk-overlay-1"]/div/div/cu-calendar-picker/div[2]/div[1]/cu-quick-date-options/button[3]/div[1]')
    tomorrow.click()
    print('set due date for tomorrow')
    time.sleep(1)
    save_task_button = driver_1.find_element(By.XPATH, '//*[@id="app-root"]/cu-app-view/cu-app-shell/cu-manager/div/div/div/main/div/div/div/div/cu-my-work-core/cu-canvas-content/cu-canvas/gridstack/gridstack-item/div/cu-canvas-card/cu-plugin/cu-my-work-plugin/cu-assigned-to-me/cu-card/div/cu-list-view-instance/cu-dashboard-table/div/div/div/cu-list-view-divisions/cu-drop-area-to-create/div/div/cu-task-list/cu-task-row-new/div/div/div[2]/button[2]')
    save_task_button.click()
    print('saved task')
    # task_config_button = driver_1.find_element(By.XPATH, '//*[@id="task-container-869e3yjnq"]/div[2]/div/div/cu-editable/cu-task-row-main/a/div')
    # task_config_button.click()
    # time.sleep(0.5)
    # description_entry = driver_1.find_element(By.XPATH, '//*[@id="app-root"]/cu-app-view/cu-task-keeper/cu-task-view/div/div[2]/div/cu-task-view-body/div/cu-task-view-task-content/div[1]/div[4]/cu-task-view-task-content-description-expanded-collapsed/div/cu-task-view-content-description/div/div[2]/cu-generate-by-prompt-blank-editor-commands/cu-generate-by-prompt-template/cu-prompt-template-empty-description/div')
    # description_entry.send_keys('This is a test')
    # exit = driver_1.find_element(By.XPATH, '//*[@id="app-root"]/cu-app-view/cu-task-keeper/cu-task-view/div/div[1]')
    # exit.click()
    time.sleep(1)
    driver_1.quit()