import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust as per your CPU cores

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime

# === Load & Preprocess Dataset ===
df = pd.read_csv('creditcard.csv')

# Scale Time and Amount
scaler_time = StandardScaler()
scaler_amount = StandardScaler()
df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1,1))
df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1,1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Rearrange columns to put scaled_time, scaled_amount first
cols = ['scaled_time', 'scaled_amount'] + [c for c in df.columns if c.startswith('V')] + ['Class']
df = df[cols]

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_res, y_res)

# === Globals & Data storage ===
history_df = pd.DataFrame(columns=['Timestamp', 'Risk', 'Decision'])
history_df['Risk'] = history_df['Risk'].astype(float)
risk_threshold = 0.7  # default threshold

# === GUI Setup ===
root = tk.Tk()
root.title("Credit Card Fraud Detection Dashboard")
root.geometry("1300x750")

# Left Frame - Form
left_frame = tk.Frame(root, width=650)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Right Frame - Dashboard
right_frame = tk.Frame(root, width=650, bg="#f0f0f0")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# --- Scrollable Form on left ---
canvas = tk.Canvas(left_frame)
scrollbar = tk.Scrollbar(left_frame, orient=tk.VERTICAL, command=canvas.yview)
form_frame = tk.Frame(canvas)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.create_window((0,0), window=form_frame, anchor="nw")

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
form_frame.bind("<Configure>", on_frame_configure)

labels = ['scaled_time', 'scaled_amount'] + [c for c in X.columns if c.startswith('V')]
entries = {}

tk.Label(form_frame, text="Enter Transaction Details", font=("Arial", 16, "bold")).pack(pady=15)

def focus_next(event):
    event.widget.tk_focusNext().focus()
    return "break"

for label in labels:
    row = tk.Frame(form_frame)
    row.pack(pady=3, padx=10, anchor='w')
    tk.Label(row, text=label+":", width=20, anchor='w').pack(side=tk.LEFT)
    e = tk.Entry(row, width=30)
    e.pack(side=tk.LEFT)
    e.bind("<Return>", focus_next)
    entries[label] = e

# --- Risk Threshold Slider ---
threshold_frame = tk.Frame(form_frame)
threshold_frame.pack(pady=10, padx=10, anchor='w')
tk.Label(threshold_frame, text="Fraud Risk Threshold:", font=("Arial", 12)).pack(side=tk.LEFT)

threshold_var = tk.DoubleVar(value=risk_threshold)
threshold_slider = tk.Scale(threshold_frame, from_=0.0, to=1.0, resolution=0.01,
                            orient=tk.HORIZONTAL, length=250, variable=threshold_var)
threshold_slider.pack(side=tk.LEFT, padx=10)

# --- Prediction Button ---
def predict():
    global history_df, risk_threshold
    input_data = []
    for label in labels:
        val = entries[label].get().strip()
        if val == '':
            messagebox.showerror("Input Error", f"Please enter a value for {label}.")
            return
        try:
            fval = float(val)
        except ValueError:
            messagebox.showerror("Input Error", f"Please enter a numeric value for {label}.")
            return
        input_data.append(fval)

    input_arr = np.array(input_data).reshape(1, -1)
    risk = model.predict_proba(input_arr)[0][1]
    risk_threshold = threshold_var.get()
    decision = "Fraud" if risk > risk_threshold else "Legit"

    if np.isnan(risk):
        messagebox.showerror("Prediction Error", "Risk score is NaN, cannot add entry.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_df.loc[len(history_df)] = [timestamp, risk, decision]

    update_dashboard()

    if risk > 0.9:
        messagebox.showwarning("High Risk Alert!", f"High risk transaction detected!\nRisk Score: {risk:.2f}")

    messagebox.showinfo("Prediction Result",
                        f"Risk Score: {risk:.3f}\nDecision: {decision}")

predict_btn = tk.Button(form_frame, text="Predict", font=("Arial", 14), bg="green", fg="white", command=predict)
predict_btn.pack(pady=15)

# --- Right frame contents ---

# Fraud/Legit Counts
count_frame = tk.Frame(right_frame, bg="#f0f0f0")
count_frame.pack(pady=10)

fraud_count_label = tk.Label(count_frame, text="Fraudulent: 0", font=("Arial", 14), fg="red", bg="#f0f0f0")
fraud_count_label.pack(side=tk.LEFT, padx=20)
legit_count_label = tk.Label(count_frame, text="Legitimate: 0", font=("Arial", 14), fg="green", bg="#f0f0f0")
legit_count_label.pack(side=tk.LEFT, padx=20)

# Matplotlib Figure for Pie and Line Charts
fig, (ax_pie, ax_line) = plt.subplots(2, 1, figsize=(5, 7))
plt.subplots_adjust(hspace=0.5)
canvas_fig = FigureCanvasTkAgg(fig, master=right_frame)
canvas_fig.get_tk_widget().pack(pady=10)

# Transaction Log Table
log_frame = tk.Frame(right_frame)
log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

columns = ("Timestamp", "Risk", "Decision")
tree = ttk.Treeview(log_frame, columns=columns, show="headings", height=12)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=150, anchor=tk.CENTER)
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar_tree = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscrollcommand=scrollbar_tree.set)
scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)

# --- Dashboard update function ---
def update_dashboard():
    fraud_count = len(history_df[history_df['Decision'] == 'Fraud'])
    legit_count = len(history_df[history_df['Decision'] == 'Legit'])

    fraud_count_label.config(text=f"Fraudulent: {fraud_count}")
    legit_count_label.config(text=f"Legitimate: {legit_count}")

    ax_pie.clear()

    # Check for valid counts to avoid NaN issues in pie chart
    if fraud_count + legit_count > 0:
        ax_pie.pie([fraud_count, legit_count], labels=['Fraud', 'Legit'],
                   colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
        ax_pie.set_title("Transaction Decisions Distribution")
    else:
        ax_pie.text(0.5, 0.5, 'No transaction data available',
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax_pie.set_title("Transaction Decisions Distribution")

    ax_line.clear()
    if not history_df.empty:
        df_plot = history_df.dropna(subset=['Risk'])
        if not df_plot.empty:
            times = pd.to_datetime(df_plot['Timestamp'])
            risks = df_plot['Risk']
            ax_line.plot(times, risks, color='blue', marker='o', linestyle='-')
            ax_line.axhline(y=risk_threshold, color='red', linestyle='--', label='Risk Threshold')
            ax_line.set_title("Risk Score Over Time")
            ax_line.set_xlabel("Timestamp")
            ax_line.set_ylabel("Risk Score")
            ax_line.legend()
            plt.setp(ax_line.get_xticklabels(), rotation=45, ha="right")
        else:
            ax_line.text(0.5, 0.5, 'No valid risk data to plot',
                         horizontalalignment='center', verticalalignment='center')
    else:
        ax_line.text(0.5, 0.5, 'No Data Yet',
                     horizontalalignment='center', verticalalignment='center')

    canvas_fig.draw()

    for item in tree.get_children():
        tree.delete(item)
    for _, row in history_df.tail(20)[::-1].iterrows():
        tree.insert("", tk.END, values=(row['Timestamp'], f"{row['Risk']:.3f}", row['Decision']))

# Start with empty dashboard
update_dashboard()

root.mainloop()
