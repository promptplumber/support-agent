"""
Builds a fake past-tickets database for the Week 2 `search_tickets` tool.

We invent ~50 realistic Nimbus support tickets. Some were resolved by support;
a few were escalated (hard edge cases) so the agent's escalate path has realistic
precedent to point at. Fake on purpose: clean, legal, fully under our control.

Run:  python scripts/build_tickets.py
Output: data/tickets.json
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data"

# id, subject, body (customer's words), category, status, resolution
TICKETS = [
    ("T-1001", "Can't log in after password change", "I changed my password and now the app says account not found.", "login", "resolved", "Customer was on the US login page but their workspace is in the EU region. Directed them to the EU login."),
    ("T-1002", "Reset link expired", "The password reset email link says it is no longer valid.", "login", "resolved", "Reset links expire after 60 minutes. Sent a fresh link and they reset successfully."),
    ("T-1003", "Locked out after too many tries", "I tried my password a bunch of times and now I'm locked out.", "login", "resolved", "Account auto-locks for 15 minutes after 5 failed attempts. Advised them to wait and use Forgot Password."),
    ("T-1004", "SSO user has no password", "Forgot password isn't working for me at all.", "login", "resolved", "Account was created via Google SSO, so it has no Nimbus password. Told them to sign in with Google."),
    ("T-1005", "Verification email never arrived", "Signed up but never got the verification email.", "account", "resolved", "Email was in spam. Also had them use Resend verification. Verified within the hour."),
    ("T-1006", "How many members on the free plan", "How many people can I add on the free plan?", "plans", "resolved", "Starter (free) allows up to 3 members and 2 active projects."),
    ("T-1007", "Team plan price", "What does the Team plan cost per person?", "plans", "resolved", "Team is 9 USD per member per month, or two months free on annual billing."),
    ("T-1008", "Charged for a member who left", "Someone left the team but I think I was still charged for them.", "billing", "escalated", "Billing is per active member during the period. Prorated credit required a manual review by the billing team."),
    ("T-1009", "Need a tax ID on my invoice", "Can you add our company VAT number to the invoice?", "billing", "resolved", "Had them add tax ID under Billing details before the next charge. Already-issued invoices can't be edited."),
    ("T-1010", "Where are my invoices", "Where can I download past invoices?", "billing", "resolved", "Settings > Billing > Invoices, downloadable as PDF. Only Owners/Admins can see billing."),
    ("T-1011", "Refund after 20 days", "I want a refund but it's been about 20 days since I paid.", "refunds", "escalated", "Outside the 14-day refund window. Escalated to support lead for a goodwill-refund decision."),
    ("T-1012", "Refund within a week", "I upgraded 5 days ago and want to undo it and get my money back.", "refunds", "resolved", "Within the 14-day window. Refund issued to original payment method, 5-10 business days."),
    ("T-1013", "Card was declined", "My payment failed and I got an email about it.", "billing", "resolved", "Failed charges retry 3 times over 5 days. Customer updated their card and the retry succeeded."),
    ("T-1014", "Downgrade blocked", "Trying to move to the free plan but it won't let me.", "plans", "resolved", "Workspace had 4 members and 3 projects, over Starter limits. Removed a member and archived a project, then downgrade worked."),
    ("T-1015", "Cancel but keep data", "If I cancel, do I lose everything?", "billing", "resolved", "Cancelling stops future charges and drops to Starter at period end. Data is kept; projects over the limit go read-only."),
    ("T-1016", "Who can cancel", "One of my admins tried to cancel and couldn't.", "billing", "resolved", "Only the Owner can cancel a subscription. Had the Owner do it."),
    ("T-1017", "Invite link expired", "The invite I sent a teammate says expired.", "members", "resolved", "Invite links are valid for 7 days. Resent the invite."),
    ("T-1018", "Bulk invite", "Is there a way to invite 15 people at once?", "members", "resolved", "Paste multiple emails separated by commas on the Invite screen."),
    ("T-1019", "Change someone's role", "How do I make a member an admin?", "members", "resolved", "Settings > Members, change their role. Explained the four roles."),
    ("T-1020", "Transfer ownership", "I'm leaving the company and need to hand over the workspace.", "members", "resolved", "Current Owner used Make Owner on an Admin. Previous Owner became an Admin. Billing moved with it."),
    ("T-1021", "Guest can't edit", "Our external contractor can only view, not edit.", "permissions", "resolved", "Guests are view-and-comment only. Changed them to Member to allow editing."),
    ("T-1022", "Archive vs delete a project", "What's the difference between archiving and deleting a project?", "projects", "resolved", "Archive keeps data and frees a plan slot; delete is permanent after a 7-day grace period."),
    ("T-1023", "Restore a deleted project", "I deleted a project by mistake yesterday.", "projects", "resolved", "Within the 7-day grace period. Restored from Projects > Recently deleted."),
    ("T-1024", "Project won't delete", "The delete option is greyed out for a member.", "projects", "resolved", "Only Owners and Admins can delete a project. An Admin did it."),
    ("T-1025", "Assign a task to someone", "I can't assign a task to a teammate.", "tasks", "resolved", "Assignee must have access to that project. Added them to the project first."),
    ("T-1026", "Subtasks progress", "Does the parent task show subtask progress?", "tasks", "resolved", "Yes, the parent shows a progress bar based on completed subtasks."),
    ("T-1027", "Dependencies missing", "I don't see the option to mark a task blocked by another.", "tasks", "resolved", "Dependencies are on Team and Business plans. Customer was on Starter."),
    ("T-1028", "Timer left running overnight", "I forgot to stop my timer and it logged 14 hours.", "time", "resolved", "Stopped the timer, then edited the entry's duration to the correct value."),
    ("T-1029", "Two timers at once", "Can I run more than one timer at the same time?", "time", "resolved", "Only one timer runs at a time; starting a new one stops the previous."),
    ("T-1030", "Edit someone else's time", "As an admin can I fix a team member's time entry?", "time", "resolved", "Yes, Admins can edit any member's entries from the task Time tab."),
    ("T-1031", "Approve timesheets", "How do I approve my team's week?", "time", "resolved", "Enable approvals in Settings > Time, then click Approve on a week. Approved weeks lock entries."),
    ("T-1032", "Export a report to CSV", "I need to get our time report into a spreadsheet.", "reports", "resolved", "Use Export on the report for CSV. Large exports are emailed as a link that expires in 24 hours."),
    ("T-1033", "Reports not showing", "I don't have a Reports tab.", "reports", "resolved", "Reports are on Team and Business plans. Customer was on Starter."),
    ("T-1034", "Slack notifications", "Can I get task updates in Slack?", "integrations", "resolved", "Connected Slack in Settings > Integrations, chose a channel and events. Also mentioned the /nimbus command."),
    ("T-1035", "Slack disconnected tasks?", "If I remove Slack do I lose my tasks?", "integrations", "resolved", "No. Disconnecting Slack only stops notifications; tasks are untouched."),
    ("T-1036", "Calendar not updating", "My Google Calendar events look out of date.", "integrations", "resolved", "Sync runs every 15 minutes and is one-way. Used Resync now to force an update."),
    ("T-1037", "Webhook signature", "How do I verify your webhooks are really from you?", "integrations", "resolved", "Generate a signing secret in Settings > Integrations > Webhooks and verify the signature on their endpoint."),
    ("T-1038", "Zapier triggers", "What can I trigger a Zap on?", "integrations", "resolved", "Triggers: task created, task completed, time entry added. Actions: create task, add time entry. Team/Business only."),
    ("T-1039", "Get an API key", "Where do I create an API key?", "api", "resolved", "Settings > Developer > API keys (Team/Business). Key inherits the creator's permissions."),
    ("T-1040", "Getting 429 from the API", "My script keeps getting 429 errors.", "api", "resolved", "Hit the rate limit (120/min Team, 600/min Business). Advised honoring the Retry-After header and batching reads."),
    ("T-1041", "Revoke a leaked key", "I think an API key got exposed in a repo.", "api", "escalated", "Advised immediate revoke from Settings > Developer. Escalated to security to review access logs for misuse."),
    ("T-1042", "Mobile app offline", "Does the app work with no signal?", "mobile", "resolved", "Recent tasks are viewable offline; changes sync when back online."),
    ("T-1043", "Mobile data out of date", "The app shows old data compared to the web.", "mobile", "resolved", "Pull to refresh, confirm correct workspace, then Settings > Sync > Resync in the app."),
    ("T-1044", "Offline change conflict", "I edited a task offline and someone else changed it too.", "mobile", "resolved", "Server version wins; the offline change is kept as a comment so nothing is lost."),
    ("T-1045", "Turn on 2FA", "How do I add two-factor authentication?", "security", "resolved", "Settings > Security > 2FA with an authenticator app. Told them to save the backup codes."),
    ("T-1046", "Locked out with 2FA", "I lost my phone and can't get past the 2FA code.", "security", "escalated", "No backup codes saved. Escalated so an Admin could reset the user's 2FA after identity check."),
    ("T-1047", "GDPR data request", "We need a copy of our data for a compliance request.", "privacy", "escalated", "Owner used Settings > Privacy > Data request. Escalated to privacy team to fulfill within the legal window."),
    ("T-1048", "Change data region", "Can we move our workspace from US to EU?", "privacy", "escalated", "Data region is fixed at creation and can't be changed. Escalated to discuss a migration path."),
    ("T-1049", "Delete our workspace", "We want to close our account and delete everything.", "privacy", "resolved", "Owner used Settings > Advanced > Delete workspace. 30-day grace period; must cancel the paid plan first."),
    ("T-1050", "App very slow", "Nimbus is crawling for our whole team today.", "troubleshooting", "resolved", "Checked status page (all green), then had them archive thousands of completed tasks in one huge project, which sped it up."),
    ("T-1051", "Import from Trello", "Can we bring our Trello boards in?", "import", "resolved", "Used the guided Trello importer; boards and lists map to projects and sections."),
    ("T-1052", "CSV import columns", "What columns does the CSV import need?", "import", "resolved", "At minimum a task title column; optional assignee email, due date (YYYY-MM-DD), status, project."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": t[0], "subject": t[1], "body": t[2], "category": t[3],
         "status": t[4], "resolution": t[5]}
        for t in TICKETS
    ]
    (OUT / "tickets.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    escalated = sum(1 for r in rows if r["status"] == "escalated")
    print(f"Wrote {len(rows)} tickets ({escalated} escalated) to {OUT / 'tickets.json'}")


if __name__ == "__main__":
    main()
