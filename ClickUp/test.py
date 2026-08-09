from selenium import webdriver
from selenium.webdriver.common.by import By
import time, sys
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium_stealth import stealth
from pathlib import Path
tasks = {
  "Morning Mathematics Block - G1C, G3M and AoPS Precalculus": {
    "due_date": "2026-08-04T08:00:00",
    "duration_minutes": 110,
    "priority": "High",
    "subject": "Mathematics",
    "description": "Complete the next ThinkAcademy G1C, ThinkAcademy Mastery G3M and AoPS Precalculus sections. Write two polished solutions that another student could follow without explanation. Mark and correct all work."
  },

  "English - Comparative Analytical Writing": {
    "due_date": "2026-08-04T09:50:00",
    "duration_minutes": 45,
    "priority": "High",
    "subject": "English",
    "description": "Choose two short extracts or moments from the same text. Write two comparative paragraphs that connect method, effect, deeper interpretation and writer's purpose. Redraft the weaker paragraph after marking it."
  },

  "French - Fluency Drill and Corrected Repeat": {
    "due_date": "2026-08-04T10:35:00",
    "duration_minutes": 20,
    "priority": "Medium",
    "subject": "French",
    "description": "Listen to a short clip twice, shadow it once and give a two-minute spoken summary without reading. Listen back, correct pauses and sentence structure, then repeat it more smoothly."
  },

  "Reading - Three Daily Sessions": {
    "due_date": "2026-08-04T12:00:00",
    "duration_minutes": 90,
    "priority": "Low",
    "subject": "Reading",
    "description": "Read for 30 minutes after breakfast, 30 minutes after lunch and 30 minutes after dinner. Choose one strong sentence from the day's reading and explain why it is effective or informative."
  },

  "Racket Sensor - Feature Comparison and Evidence Plots": {
    "due_date": "2026-08-04T14:00:00",
    "duration_minutes": 50,
    "priority": "High",
    "subject": "Design Engineering",
    "description": "Using labelled data, compare at least three features such as peak acceleration, angular velocity, event duration or signal area. Create clear plots or a comparison table and select the most useful feature with evidence."
  },

  "Cello - Rhythm and Intonation": {
    "due_date": "2026-08-04T19:30:00",
    "duration_minutes": 30,
    "priority": "Medium",
    "subject": "Cello",
    "description": "Practise one scale slowly with a tuner or drone, then use a metronome on the current piece. Isolate an unstable passage and play it accurately three times before continuing."
  }
}
for i in tasks.keys():
    automation_profile = Path.home() / "selenium-chrome-profile"
    automation_profile.mkdir(exist_ok=True)

    options = webdriver.ChromeOptions()
    options.add_argument(f"--user-data-dir={automation_profile}")
    options.add_argument("--profile-directory=Default")
    options.add_experimental_option('detach', True)
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