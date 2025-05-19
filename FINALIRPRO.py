import time
import datetime
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.font import Font
from PIL import Image, ImageTk
import threading
from bs4 import BeautifulSoup
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import google.generativeai as genai
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sentence_transformers import SentenceTransformer, util
# --- Configure Gemini ---
genai.configure(api_key="AIzaSyCkQ5UBlkbzE7uHqm0QfX0F0vWQ5wZg60w")
model = genai.GenerativeModel("gemini-1.5-flash")
sem_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_relevance_score(title, query, threshold=0.5):
    try:
        embeddings = sem_model.encode([title, query], convert_to_tensor=True)
        score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return score >= threshold
    except Exception as e:
        print(f"⚠️ Semantic check failed: {e}")
        return False
# --- Initialize Selenium Driver ---
def init_driver():
    ua = UserAgent()
    options = Options()
    options.add_argument(f"user-agent={ua.random}")
    options.add_argument("--start-maximized")
    service = Service(r"C:\\Users\\rathi\\Downloads\\chromedriver-win64\\chromedriver.exe")
    return webdriver.Chrome(service=service, options=options)

# --- Clean Amazon URL ---
def clean_amazon_url(url):
    match = re.search(r'/dp/([A-Z0-9]{10})', url)
    return f"https://www.amazon.in/dp/{match.group(1)}" if match else "N/A"

# --- Sentiment Analysis ---
def get_sentiment(rating, reviews):
    try:
        rating = float(rating)
        reviews = int(reviews)
    except:
        return "Uncertain ❓"

    if rating >= 4.3:
        base = "Positive"
    elif 3.5 <= rating < 4.3:
        base = "Neutral"
    else:
        base = "Negative"

    if reviews >= 100:
        confidence = "Highly"
    elif reviews >= 50:
        confidence = "Moderately"
    elif reviews >= 20:
        confidence = "Slightly"
    else:
        return "Uncertain ❓"

    emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
    return f"{confidence} {base} {emoji[base]}"
# --- Update Price History ---
def update_price_history(df, history_file="price_history.csv"):
    df['Timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        old_df = pd.read_csv(history_file)
        df_all = pd.concat([old_df, df], ignore_index=True)
    except FileNotFoundError:
        df_all = df
    df_all.to_csv(history_file, index=False)
    print(f"📈 Price history updated in '{history_file}'")

# --- Plot Price History Graph ---
def plot_price_history(product_title, history_file="price_history.csv"):
    df = pd.read_csv(history_file)
    df_product = df[df['Title'] == product_title].copy()

    if df_product.empty:
        print(f"No price history for '{product_title}'.")
        return None

    df_product['Timestamp'] = pd.to_datetime(df_product['Timestamp'])
    df_product = df_product.sort_values('Timestamp')

    plt.figure(figsize=(10, 5))
    plt.plot(df_product['Timestamp'], df_product['Price'], marker='o', linestyle='-', linewidth=2)
    plt.title(f"Price Tracker: {product_title}", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (INR)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"{product_title[:20].replace(' ','_').replace('.','_')}_price_{timestamp}.png"
    plt.savefig(img_filename)
    plt.close()
    print(f"📊 Price graph saved as '{img_filename}'")
    return img_filename

# --- Fetch Amazon Products ---
def fetch_amazon_products(query, page_limit=1):
    driver = init_driver()
    driver.get("https://www.amazon.in")
    wait = WebDriverWait(driver, 15)
    time.sleep(random.uniform(2, 4))

    while "captcha" in driver.page_source.lower():
        messagebox.showwarning("CAPTCHA Detected", "Please solve the CAPTCHA on Amazon manually.")
        time.sleep(10)

    try:
        search_box = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@id='twotabsearchtextbox']")))
        search_box.clear()
        search_box.send_keys(query)
        search_box.submit()
    except Exception as e:
        print(f"Amazon search failed: {e}")
        driver.quit()
        return pd.DataFrame()

    all_products = []
    for _ in range(page_limit):
        soup = BeautifulSoup(driver.page_source, "lxml")
        results = soup.find_all("div", {"data-component-type": "s-search-result"})

        for item in results:
            try:
                title = item.h2.text.strip()
                link = clean_amazon_url(f"https://www.amazon.in{item.find('a')['href']}")
                price = item.select_one("span.a-price > span.a-offscreen").text.replace("₹", "").replace(",", "")
                rating = item.select_one("span.a-icon-alt").text.split()[0]
                reviews = item.select_one("div.a-row.a-size-small > a > span").text.replace(",", "")
                sentiment = get_sentiment(rating, reviews)
                all_products.append({
                    "Platform": "Amazon",
                    "Title": title,
                    "Link": link,
                    "Price": float(price),
                    "Rating": float(rating),
                    "Reviews": int(reviews),
                    "Availability": "Available",
                    "Sentiment": sentiment
                })
            except:
                continue

        try:
            driver.find_element(By.CSS_SELECTOR, "a.s-pagination-next").click()
            time.sleep(random.uniform(3, 5))
        except:
            break

    driver.quit()
    return pd.DataFrame(all_products)

# --- Fetch Flipkart Products ---
def scrape_flipkart_data(search_query, pages=3):
    driver = init_driver()
    driver.get("https://www.flipkart.com")
    time.sleep(2)

    try:
        driver.find_element(By.XPATH, "//button[contains(text(),'✕')]").click()
    except:
        pass

    search_input = driver.find_element(By.NAME, "q")
    search_input.send_keys(search_query)
    search_input.submit()
    time.sleep(3)

    products = []

    for page in range(pages):
        time.sleep(2)

        # Layout Handling
        if driver.find_elements(By.XPATH, "//div/div[@class='KzDlHZ']"):
            title_elements = driver.find_elements(By.XPATH, "//div/div[@class='KzDlHZ']")
            price_elements = driver.find_elements(By.XPATH, "//div/div[@class='Nx9bqj _4b5DiR']")
            rating_elements = driver.find_elements(By.XPATH, "//span/div[@class='XQDdHH']")
            reviews_elements = driver.find_elements(By.XPATH, "//span[contains(text(),'Reviews')]")
        else:
            title_elements = driver.find_elements(By.CLASS_NAME, "wjcEIp")
            price_elements = driver.find_elements(By.XPATH, "//div/div[@class='Nx9bqj']")
            rating_elements = driver.find_elements(By.XPATH, "//span/div[@class='XQDdHH']")
            reviews_elements = driver.find_elements(By.XPATH, "//div/span[@class='Wphh3N']")

        max_len = max(len(title_elements), len(price_elements), len(rating_elements), len(reviews_elements))

        for idx in range(max_len):
            try:
                title = title_elements[idx].text if idx < len(title_elements) else "N/A"
                price = price_elements[idx].text.replace("₹", "").replace(",", "") if idx < len(price_elements) else "0"
                rating = rating_elements[idx].text if idx < len(rating_elements) else "0"

                if idx < len(reviews_elements):
                    review_raw = reviews_elements[idx].text.strip()
                    if "Reviews" in review_raw:
                        reviews = re.sub(r"[^\d]", "", review_raw)
                    else:
                        reviews = review_raw.replace("(", "").replace(")", "")
                else:
                    reviews = "0"

                # ✅ Get correct product link
                try:
                    a_tag = title_elements[idx].find_element(By.XPATH, ".//ancestor::a")
                    raw_link = a_tag.get_attribute("href")
                    link = "https://www.flipkart.com" + raw_link if raw_link.startswith("/") else raw_link
                except:
                    link = driver.current_url

                sentiment = get_sentiment(rating, reviews)

                products.append({
                    "Platform": "Flipkart",
                    "Title": title,
                    "Link": link,
                    "Price": float(price),
                    "Rating": float(rating),
                    "Reviews": int(reviews),
                    "Availability": "Available",
                    "Sentiment": sentiment
                })
            except:
                continue

        # ✅ Pagination using numbered buttons
        try:
            pagination = driver.find_elements(By.XPATH, '//nav/a[@class="cn++Ap"]')
            if page < len(pagination):
                pagination[page].click()
                print(f"➡️ Clicked page {page + 2}")
                time.sleep(3)
            else:
                print("🚫 Reached last page.")
                break
        except Exception as e:
            print(f"⚠️ Pagination error: {e}")
            break

    driver.quit()
    return pd.DataFrame(products)

def fetch_flipkart_products(search_query, pages=3):
    df = scrape_flipkart_data(search_query, pages)

    if df.empty:
        print("⚠️ No Flipkart products found.")
        return df

    df['Platform'] = 'Flipkart'
    df['Link'] = f"https://www.flipkart.com/search?q={search_query}"
    df['Availability'] = 'Available'
    df['Sentiment'] = df.apply(lambda row: get_sentiment(row['Rating'], row['Reviews']), axis=1)
    df = df.rename(columns={"Product Title": "Title"})

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0)

    print(f"✅ Flipkart scraped {len(df)} products.")
    return df



# --- Calculate Scores for Products ---
def calculate_scores(df, platform_name):
    df = df.copy()
    df = df[(df['Price'] > 0) & (df['Rating'] > 0)]

    if df.empty:
        print(f"⚠️ No valid data to score for {platform_name}")
        return df  # Return empty DataFrame to avoid issues

    df['Norm_Price'] = 1 - ((df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min() + 1e-5))
    df['Norm_Rating'] = (df['Rating'] - df['Rating'].min()) / (df['Rating'].max() - df['Rating'].min() + 1e-5)
    df['Norm_Reviews'] = (df['Reviews'] - df['Reviews'].min()) / (df['Reviews'].max() - df['Reviews'].min() + 1e-5)

    df['Score'] = ((df['Norm_Price'] * 0.4) + (df['Norm_Rating'] * 0.4) + (df['Norm_Reviews'] * 0.2)) * 10
    df['Score'] = df['Score'].round(2)

    df_sorted = df.sort_values(by='Score', ascending=False)
    filename = f"{platform_name.lower()}_scored_products.csv"
    df_sorted.to_csv(filename, index=False)
    print(f"✅ Scored {platform_name} data saved to '{filename}'")
    return df_sorted


# --- Evaluate Tracker Accuracy ---
def evaluate_tracker_accuracy(df, platform_name, query=None):
    total_entries = len(df)
    if total_entries == 0:
        print(f"⚠️ No data to evaluate for {platform_name}")
        return 0

    valid_entries = df[
        (df['Price'] > 0) &
        (df['Rating'] > 0) &
        df['Title'].apply(lambda x: x != "N/A" and isinstance(x, str)) &
        df['Link'].apply(lambda x: x != "N/A" and isinstance(x, str))
    ]

    if query:
        valid_entries = valid_entries[valid_entries['Title'].apply(lambda title: semantic_relevance_score(title, query))]

    accuracy_percent = (len(valid_entries) / total_entries) * 100
    accuracy_percent = round(accuracy_percent, 2)

    print(f"📊 {platform_name} Tracker Accuracy (Semantic + Valid): {accuracy_percent}% ({len(valid_entries)}/{total_entries})")
    return accuracy_percent


# --- Filter High Score Products ---
import os

def filter_score_above_six():
    try:
        amazon_df = pd.read_csv("amazon_scored_products.csv")
        flipkart_df = pd.read_csv("flipkart_scored_products.csv")
    except FileNotFoundError:
        print("❌ Scored product files not found. Run calculate_scores() first.")
        return None

    combined_df = pd.concat([amazon_df, flipkart_df], ignore_index=True)
    high_score_df = combined_df[combined_df['Score'] > 6]

    if high_score_df.empty:
        print("ℹ️ No products with score > 6.")
        return None

    high_score_df.to_csv("high_score_above_6.csv", index=False)
    print(f"✅ High score >6 products saved to 'high_score_above_6.csv' ({len(high_score_df)} entries)")
    return high_score_df
def evaluate_high_score_accuracy_by_platform():
    try:
        df = pd.read_csv("high_score_above_6.csv")
    except FileNotFoundError:
        print("⚠️ high_score_above_6.csv not found.")
        return {}

    results = {}
    for platform in ["Amazon", "Flipkart"]:
        platform_df = df[df['Platform'] == platform]
        total = len(platform_df)
        valid = platform_df[
            (platform_df['Price'] > 0) &
            (platform_df['Rating'] > 0) &
            platform_df['Title'].apply(lambda x: x != "N/A" and isinstance(x, str)) &
            platform_df['Link'].apply(lambda x: x != "N/A" and isinstance(x, str))
        ]
        accuracy = (len(valid) / total) * 100 if total > 0 else 0
        results[platform] = round(accuracy, 2)
        print(f"📊 {platform} Score >6 Accuracy: {results[platform]}% ({len(valid)}/{total})")

    return results


# --- Ask Gemini AI for Insights ---
def ask_gemini(df, prompts, output_txt_filename):
    # Ensure Score exists
    if 'Score' not in df.columns:
        print("⚠️ Score column missing. Recalculating...")
        df = calculate_scores(df, "Combined")

    # Sort by Score and pick top 5 from each platform
    top_amazon = df[df["Platform"] == "Amazon"].sort_values(by="Score", ascending=False).head(5)
    top_flipkart = df[df["Platform"] == "Flipkart"].sort_values(by="Score", ascending=False).head(5)
    top_df = pd.concat([top_amazon, top_flipkart])

    # Format table to include all useful fields
    table_text = top_df[["Platform", "Title", "Price", "Rating", "Reviews", "Score"]].to_string(index=False)

    with open(output_txt_filename, "w", encoding="utf-8") as f:
        for prompt in prompts:
            full_prompt = (
                f"{prompt}\n\n"
                "Here are the top 5 products each from Amazon and Flipkart ranked by Score.\n"
                "All prices are in Indian Rupees (₹).\n\n"
                f"{table_text}\n\n"
                "Please:\n"
                "- Identify  ten similar or same items across platforms\n"
                "- Compare Score, Rating, and Price to suggest best value\n"
                "- Recommend 1 product from each platform (or cross-platform) for the customer\n"
                "- Use emoji to make it engaging\n"
                "- Keep it short and easy to read\n"
            )

            try:
                response = model.generate_content(full_prompt)
                print(f"\n🤖 Gemini Says ({prompt}):\n")
                print(response.text)
                f.write(f"\n--- Prompt: {prompt} ---\n{response.text}\n\n")
            except Exception as e:
                print(f"⚠️ Gemini request failed: {e}")
                f.write(f"\n--- Prompt: {prompt} ---\nError: {e}\n\n")

# --- Send Email Report ---
def send_email(to_email, csv_file, txt_file, image_files=None):
    sender_email = "rathi.achintya@gmail.com"
    sender_password = "kqta nkgm uolm asfp"  # Your Google app password here

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = "📦 Product Report + Gemini Insights"

    msg.attach(MIMEText("Attached are your product report and AI analysis.", 'plain'))

    attachments = [
                      "amazon_scored_products.csv",
                      "flipkart_scored_products.csv",
                      csv_file,
                      txt_file,
                      "high_score_above_6.csv"  # <- Add this here
                  ] + (image_files or [])

    for file_path in attachments:
        try:
            with open(file_path, "rb") as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{file_path}"')
                msg.attach(part)
        except Exception as e:
            print(f"⚠️ Failed to attach {file_path}: {e}")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print(f"📩 Email sent successfully to {to_email}!")
        return True
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False

class PriceTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Amazon & Flipkart Price Tracker")
        self.geometry("1200x800")
        self.configure(bg="#f5f5f5")

        # Header, Search, Results, Graph & Footer
        self.create_header()
        self.create_search_section()
        self.create_results_section()
        self.create_graph_section()
        self.create_footer()

    def create_header(self):
        header = tk.Frame(self, bg="#2c3e50", height=70)
        header.pack(fill=tk.X)
        tk.Label(header, text="Amazon & Flipkart Price Tracker", font=("Helvetica", 24, "bold"),
                 fg="white", bg="#2c3e50").pack(pady=15)

    def create_search_section(self):
        frame = tk.Frame(self, bg="#f5f5f5", pady=20)
        frame.pack(fill=tk.X)
        tk.Label(frame, text="Enter Product Name:", font=("Helvetica", 12), bg="#f5f5f5").pack(side=tk.LEFT, padx=20)
        self.search_entry = tk.Entry(frame, font=("Helvetica", 12), width=40)
        self.search_entry.pack(side=tk.LEFT, padx=10)
        tk.Button(frame, text="Search", command=self.search_products, font=("Helvetica", 12),
                  bg="#3498db", fg="white", padx=15).pack(side=tk.LEFT, padx=10)

    def create_results_section(self):
        frame = tk.Frame(self, bg="white", padx=20, pady=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.amazon_tab, self.flipkart_tab, self.combined_tab = tk.Frame(self.notebook), tk.Frame(self.notebook), tk.Frame(self.notebook)
        self.notebook.add(self.amazon_tab, text="Amazon")
        self.notebook.add(self.flipkart_tab, text="Flipkart")
        self.notebook.add(self.combined_tab, text="Combined")

        self.create_treeview(self.amazon_tab, "amazon_tree")
        self.create_treeview(self.flipkart_tab, "flipkart_tree")
        self.create_treeview(self.combined_tab, "combined_tree")

    def create_treeview(self, parent, name):
        columns = ("Platform", "Title", "Price", "Rating", "Reviews", "Score", "Sentiment")
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        setattr(self, name, tree)
        for col in columns:
            tree.heading(col, text=col)
            width = 100 if col != "Title" else 300
            tree.column(col, width=width)
        tree.pack(fill=tk.BOTH, expand=True)
        y_scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.bind("<<TreeviewSelect>>", self.on_product_select)

    def create_graph_section(self):
        frame = tk.Frame(self, bg="white", height=300)
        frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(frame, text="Price History Graph", font=("Helvetica", 14, "bold"), bg="white").pack(pady=10)
        self.graph_canvas_frame = tk.Frame(frame, bg="white")
        self.graph_canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.graph_placeholder = tk.Label(self.graph_canvas_frame, text="Select a product to view price history",
                                          font=("Helvetica", 12), bg="white")
        self.graph_placeholder.pack(pady=100)

    def create_footer(self):
        footer = tk.Frame(self, bg="#2c3e50", height=30)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(footer, text="© 2025 Price Tracker Tool | BITS Pilani", font=("Helvetica", 10),
                 fg="white", bg="#2c3e50").pack(pady=5)
        tk.Button(footer, text="Save Score > 6", font=("Helvetica", 10),
                  command=self.save_high_score_products,
                  bg="#e67e22", fg="white", padx=10).pack(side=tk.RIGHT, padx=10)

    def search_products(self):
        query = self.search_entry.get()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a product name.")
            return

        progress = tk.Toplevel(self)
        progress.title("Searching...")
        progress.geometry("400x150")
        tk.Label(progress, text=f"Searching for '{query}'...\nPlease wait.", font=("Helvetica", 12)).pack(pady=20)
        bar = ttk.Progressbar(progress, orient="horizontal", length=350, mode="indeterminate")
        bar.pack(pady=10)
        bar.start()

        for tree_name in ["amazon_tree", "flipkart_tree", "combined_tree"]:
            tree = getattr(self, tree_name)
            for item in tree.get_children():
                tree.delete(item)

        threading.Thread(target=lambda: self.perform_search(query, progress), daemon=True).start()

    def save_high_score_products(self):
        df = filter_score_above_six()
        if df is not None:
            messagebox.showinfo("Saved", f"Saved {len(df)} high-score products to 'high_score_above_6.csv'")
        else:
            messagebox.showinfo("No Products", "No high-score products to save.")

    def perform_search(self, query, progress_window):
        try:
            df_amazon = fetch_amazon_products(query, page_limit=2)
            df_flipkart = fetch_flipkart_products(query, pages=2)

            if not df_amazon.empty:
                calculate_scores(df_amazon, "Amazon")
                evaluate_tracker_accuracy(df_amazon, "Amazon")

            if not df_flipkart.empty:
                calculate_scores(df_flipkart, "Flipkart")
                evaluate_tracker_accuracy(df_flipkart, "Flipkart")

            df_combined = pd.concat([df_amazon, df_flipkart], ignore_index=True)
            if df_combined.empty:
                self.after(100, lambda: [progress_window.destroy(),
                                         messagebox.showinfo("No Results", "No products found.")])
                return

            update_price_history(df_combined)
            df_combined.to_csv("combined_products.csv", index=False)
            self.after(100, lambda: self.update_results(df_amazon, df_flipkart, df_combined, progress_window))

        except Exception as e:
            self.after(100, lambda err=e: [progress_window.destroy(),
                                           messagebox.showerror("Error", f"An error occurred: {str(err)}")])

    def update_results(self, df_amazon, df_flipkart, df_combined, progress_window):
        self.populate_treeview(self.amazon_tree, df_amazon)
        self.populate_treeview(self.flipkart_tree, df_flipkart)
        self.populate_treeview(self.combined_tree, df_combined)
        self.notebook.select(2)

        progress_window.destroy()
        messagebox.showinfo("Search Complete", f"Found {len(df_combined)} products.")

        if not df_combined.empty:
            first_title = df_combined.iloc[0]['Title']
            self.display_price_graph(first_title)

            # Automatically prompt for email sending after analysis:
            self.prompt_email_report(first_title)

    def populate_treeview(self, tree, df):
        for item in tree.get_children():
            tree.delete(item)
        for _, row in df.iterrows():
            values = (
                row.get('Platform', ''),
                row.get('Title', ''),
                f"₹{row.get('Price', 0):.2f}",
                f"{row.get('Rating', 0):.1f}",
                row.get('Reviews', 0),
                f"{row.get('Score', 0):.2f}" if 'Score' in row else '',
                row.get('Sentiment', '')
            )
            tree.insert('', 'end', values=values)

    def on_product_select(self, event):
        tree = event.widget
        selection = tree.selection()
        if not selection:
            return
        item = tree.item(selection[0])
        title = item['values'][1]
        self.display_price_graph(title)

    def display_price_graph(self, product_title):
        for widget in self.graph_canvas_frame.winfo_children():
            widget.destroy()

        loading = tk.Label(self.graph_canvas_frame, text="Generating graph...", font=("Helvetica", 12), bg="white")
        loading.pack(pady=50)
        self.update_idletasks()

        img_file = plot_price_history(product_title)
        loading.destroy()

        if img_file:
            img = Image.open(img_file).resize((900, 300), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            img_label = tk.Label(self.graph_canvas_frame, image=tk_img, bg="white")
            img_label.image = tk_img
            img_label.pack()

            btn_frame = tk.Frame(self.graph_canvas_frame, bg="white")
            btn_frame.pack(pady=10)

            tk.Button(btn_frame, text="Send Report to Email",
                      command=lambda: self.send_email_report(product_title, img_file),
                      font=("Helvetica", 10), bg="#27ae60", fg="white", padx=10).pack(side=tk.LEFT, padx=5)

            tk.Button(btn_frame, text="Refresh Price Data",
                      command=lambda: self.refresh_product_price(product_title),
                      font=("Helvetica", 10), bg="#3498db", fg="white", padx=10).pack(side=tk.LEFT, padx=5)
        else:
            tk.Label(self.graph_canvas_frame, text="No price history available.", font=("Helvetica", 12), bg="white").pack(pady=100)
    def refresh_product_price(self, product_title):
        try:
            df_combined = pd.read_csv("combined_products.csv")
            product_row = df_combined[df_combined['Title'] == product_title].iloc[0]
            platform = product_row['Platform']

            query = " ".join(product_title.split()[:5])  # Use first 5 words for re-search
            messagebox.showinfo("Refreshing", f"Refreshing price for:\n{product_title}")

            if platform == 'Amazon':
                df_new = fetch_amazon_products(query, page_limit=1)
            elif platform == 'Flipkart':
                df_new = fetch_flipkart_products(query, page_limit=3)
            else:
                messagebox.showwarning("Unknown Platform", "Cannot refresh for unknown platform.")
                return

            if not df_new.empty:
                update_price_history(df_new)
                self.display_price_graph(product_title)
                messagebox.showinfo("Updated", "Price history refreshed!")
            else:
                messagebox.showinfo("No Update", "No new data found for this product.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh price: {str(e)}")

    def prompt_email_report(self, product_title):
        response = messagebox.askyesno(
            "Send Report via Email",
            "Do you want to send the product analysis via email?"
        )
        if response:
            self.send_email_dialog(product_title)

    def send_email_dialog(self, product_title):
        dialog = tk.Toplevel(self)
        dialog.title("Send Email Report")
        dialog.geometry("400x200")
        tk.Label(dialog, text="Enter recipient email:", font=("Helvetica", 12), pady=10).pack()
        email_entry = tk.Entry(dialog, font=("Helvetica", 12), width=30)
        email_entry.pack(pady=5)
        email_entry.insert(0, "example@gmail.com")

        tk.Button(
            dialog,
            text="Send Report",
            command=lambda: self.auto_process_email_send(email_entry.get(), product_title, dialog),
            font=("Helvetica", 12),
            bg="#3498db",
            fg="white",
            padx=15
        ).pack(pady=20)

    def auto_process_email_send(self, email, product_title, dialog):
        if not email or "@" not in email:
            messagebox.showwarning("Invalid Email", "Enter a valid email address.")
            return
        try:
            df_combined = pd.read_csv("combined_products.csv")
            df_product = df_combined[df_combined['Title'] == product_title]
            if df_product.empty:
                messagebox.showerror("Error", "Product data not found.")
                dialog.destroy()
                return

            # Ask to save Score > 6 CSV
            if messagebox.askyesno("Save Score > 6",
                                   "Do you want to save Score > 6 products and include in the email?"):
                high_df = filter_score_above_six()
                if high_df is not None:
                    accuracy_dict = evaluate_high_score_accuracy_by_platform()
                    msg = "\n".join([f"{platform}: {acc}%" for platform, acc in accuracy_dict.items()])
                    messagebox.showinfo("High Score Accuracy", f"Tracker Accuracy for Score > 6 products:\n\n{msg}")

                else:
                    messagebox.showinfo("Info", "No products with Score > 6 were found.")

            # Prepare product CSV and Gemini insights
            temp_csv = f"{product_title[:20].replace(' ', '_')}_report.csv"
            df_product.to_csv(temp_csv, index=False)

            prompts = [
                f"Analyze this product: {product_title}",
                "Is this a good time to buy based on price trends?"
            ]
            temp_txt = f"{product_title[:20].replace(' ', '_')}_analysis.txt"
            ask_gemini(df_product, prompts, temp_txt)

            img_file = plot_price_history(product_title)

            # Send email with all attachments including Score > 6 CSV
            success = send_email(email, temp_csv, temp_txt, image_files=[img_file])
            dialog.destroy()

            if success:
                messagebox.showinfo("Success", f"Report sent to {email}")
            else:
                messagebox.showerror("Failed", "Could not send the email.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to send email: {str(e)}")
            dialog.destroy()
def ask_gemini(df, prompts, output_txt_filename):
    # Ensure Score exists
    if 'Score' not in df.columns:
        print("⚠️ Score column missing. Recalculating...")
        df = calculate_scores(df, "Combined")

    # Sort by Score and pick top 5 from each platform
    top_amazon = df[df["Platform"] == "Amazon"].sort_values(by="Score", ascending=False).head(5)
    top_flipkart = df[df["Platform"] == "Flipkart"].sort_values(by="Score", ascending=False).head(5)
    top_df = pd.concat([top_amazon, top_flipkart])

    # Format table to include all useful fields
    table_text = top_df[["Platform", "Title", "Price", "Rating", "Reviews", "Score"]].to_string(index=False)

    with open(output_txt_filename, "w", encoding="utf-8") as f:
        for prompt in prompts:
            full_prompt = (
                f"{prompt}\n\n"
                "Here are the top 5 products each from Amazon and Flipkart ranked by Score.\n"
                "All prices are in Indian Rupees (₹).\n\n"
                f"{table_text}\n\n"
                "Please:\n"
                "- Identify similar or same items across platforms\n"
                "- Compare Score, Rating, and Price to suggest best value\n"
                "- Recommend 1 product from each platform (or cross-platform) for the customer\n"
                "- Use emoji to make it engaging\n"
                "- Keep it short and easy to read\n"
            )

            try:
                response = model.generate_content(full_prompt)
                print(f"\n🤖 Gemini Says ({prompt}):\n")
                print(response.text)
                f.write(f"\n--- Prompt: {prompt} ---\n{response.text}\n\n")
            except Exception as e:
                print(f"⚠️ Gemini request failed: {e}")
                f.write(f"\n--- Prompt: {prompt} ---\nError: {e}\n\n")


# --- Main Function ---
def main():
    app = PriceTrackerApp()

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Treeview", background="#f9f9f9", foreground="black", rowheight=25, fieldbackground="#f9f9f9")
    style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'), background="#e0e0e0")
    style.map('Treeview', background=[('selected', '#3498db')])

    app.mainloop()

if __name__ == "__main__":
    main()
