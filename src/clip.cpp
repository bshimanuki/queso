#include <iostream>

#include <gtk/gtk.h>

constexpr int num = 3;

class ClipboardManager {
public:
  const char *text = "testtext";
  const char *html = "<meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\"><style type=\"text/css\"><!--td {border: 1px solid #ccc;}br {mso-data-placement:same-cell;}--></style><span style=\"font-size:10pt;font-family:Arial;font-style:normal;text-decoration:line-through;color:#ff0000;\" data-sheets-value=\"{&quot;1&quot;:2,&quot;2&quot;:&quot;c&quot;}\" data-sheets-userformat=\"{&quot;2&quot;:330241,&quot;3&quot;:{&quot;1&quot;:0},&quot;12&quot;:0,&quot;14&quot;:{&quot;1&quot;:2,&quot;2&quot;:16711680},&quot;19&quot;:1,&quot;21&quot;:0}\">c</span>";
  const char *string = "string";
  const GtkTargetEntry targets[num] = {
    {(gchar*) "text/plain", 0, 0},
    {(gchar*) "text/html", 0, 1},
    {(gchar*) "STRING", 0, 2},
  };
  const char *values[num] = {
    text,
    html,
    string,
  };
  void copy_cb(GtkClipboard *clip, GtkSelectionData *selection_data, guint info) {
    GdkAtom target = gtk_selection_data_get_target(selection_data);
    target = gdk_atom_intern(gdk_atom_name(target), false);
    gtk_selection_data_set(selection_data, target, 8, (const guchar*) values[info], strlen(values[info]));
  }
};

void copy_cb(GtkClipboard *clip, GtkSelectionData *selection_data, guint info, gpointer owner) {
  ClipboardManager *self = (ClipboardManager*) owner;
  self->copy_cb(clip, selection_data, info);
}

void clipboard_callback(GtkClipboard *clip, const gchar *text, gpointer data) {
  g_print("%s\n", text);
  gtk_main_quit();
}

void get_contents(GtkClipboard *clip) {
  GtkSelectionData *data = gtk_clipboard_wait_for_contents(clip, gdk_atom_intern("text/html", false));
  std::cout << gtk_selection_data_get_data(data) << std::endl;
  std::cout << gdk_atom_name(gtk_selection_data_get_data_type(data)) << std::endl;
  std::cout << gtk_selection_data_get_format(data) << std::endl;
}

int main(int argc, char **argv) {
  ClipboardManager manager;
  gtk_init(&argc, &argv);
  GtkClipboard *clip = gtk_clipboard_get(GDK_SELECTION_CLIPBOARD);
  // gtk_clipboard_request_text(clip, clipboard_callback, NULL);
  gtk_clipboard_set_with_data(clip, manager.targets, num, copy_cb, nullptr, &manager);
  // get_contents(clip);
  gtk_main();
  return 0;
}
