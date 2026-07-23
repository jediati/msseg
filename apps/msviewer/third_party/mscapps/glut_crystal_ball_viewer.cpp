#include "glut_crystal_ball_viewer.h"


namespace CRYSTAL_BALL_VIEWER {


	namespace INTERNAL {

		float g_specular_light[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		GLfloat g_diffuse_light[] = { 1.0, 0.8, 0.8, 1.0 };  /* Red diffuse light. */
		GLfloat g_light_position[] = { 1.0, 1.0, 1.0, 0.0 };  /* Infinite light location. */

		Matrix4D g_proj;
		Matrix4D g_modelview;
		Matrix4D g_modproj;
		Matrix4 g_rotation_matrix;
		Quaternion g_rotation_total_quat;

		// Bounds of viewing frustum.
		double g_near_plane = 1;
		double g_far_plane = 3000;

		// Viewing angle.
		double g_field_of_view = 40.0;
		Matrix4 g_base_view_matrix;
		float g_scene_center_displacement[3];
		float g_dist_from_center;

		RenderableGL* g_scene = NULL;
		bool g_mouseActive = false;
		bool g_draw_bounding_box = true;
		bool g_clear_color;


		int g_main_window;
		int g_mouse_x_old;
		int g_mouse_y_old;
		Vector4 g_screen_start_pos;
		int g_mouse_button_state;
		int g_window_width = 800;
		int g_window_height = 600;

		bool(*g_user_mouse_func)(int, int, int, int) = NULL;
		bool(*g_user_mouse_moved_func)(int, int) = NULL;
		bool(*g_user_graphic_keys_func)(unsigned char, int, int) = NULL;
		bool(*g_user_passive_mouse_func)(int, int) = NULL;

		void mouseClicked(int button, int state, int x, int y)
		{
			if (g_user_mouse_func != NULL && g_user_mouse_func(button, state, x, y)) return;
			//printf("%d, %d\n", x, y);
			g_mouse_x_old = x;
			g_mouse_y_old = y;

			//printf("Mouse Clicked! button = %d, %d  x = %d   y = %d\n", state, button, x, y);
			g_mouse_button_state = button;
			if (state == GLUT_DOWN) {
				g_mouseActive = true;
			}
			else {
				g_mouseActive = false;
			}

			glutPostRedisplay();
		}


		void Initialize_View() {


		}

		void reset_view() {
			Quaternion::resetQuat(g_rotation_total_quat);
			g_base_view_matrix = MIdentity();
			Vector4 vmin, vmax;
			g_scene->BBox(vmin, vmax);
			Vector4 lengths = (vmax - vmin);
			g_dist_from_center = 300 * lengths[2];
		}

		void render_view(){
			Quaternion::resetQuat(g_rotation_total_quat);
			g_base_view_matrix = MIdentity();
			Vector4 vmin, vmax;
			g_scene->BBox(vmin, vmax);
			Vector4 lengths = (vmax - vmin);
			g_dist_from_center = 240 * lengths[2] ;
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			Vector4 v; v.vals[0] = 0; v.vals[1] = 1; v.vals[2] = 0;
			g_rotation_total_quat = Quaternion::rotation(v, -50 * PI / 180.0);
			Quaternion::setRotQuat(g_rotation_matrix, g_rotation_total_quat);
			glMultMatrixf(g_rotation_matrix.vals);
		}

		void render_view_2(){
			printf("saving matrix...\n");
			FILE* fmat = fopen("view_matrix.txt", "w");
			fprintf(fmat, "%f ", g_dist_from_center);
			fprintf(fmat, "%f ", g_rotation_total_quat.x);
			fprintf(fmat, "%f ", g_rotation_total_quat.y);
			fprintf(fmat, "%f ", g_rotation_total_quat.z);
			fprintf(fmat, "%f ", g_rotation_total_quat.w);
			for (int i = 0; i < 16; i++)
				fprintf(fmat, "%f ", g_base_view_matrix.vals[i]);
			fflush(fmat);
			fclose(fmat);
		}
		void render_view_3(){
			printf("loading matrix...\n");
			FILE* fmat = fopen("view_matrix.txt", "r");
			fscanf(fmat, "%f ", &g_dist_from_center);
			fscanf(fmat, "%f ", &g_rotation_total_quat.x);
			fscanf(fmat, "%f ", &g_rotation_total_quat.y);
			fscanf(fmat, "%f ", &g_rotation_total_quat.z);
			fscanf(fmat, "%f ", &g_rotation_total_quat.w);
			for (int i = 0; i < 16; i++)
				fscanf(fmat, "%f ", &g_base_view_matrix.vals[i]);
			fclose(fmat);
			glutPostRedisplay();
		}


		void display() {


			glPointSize(4.0);
			if (g_clear_color) {
				glClearColor(1, 1, 1, 1);
			}
			else {
				glClearColor(0, 0, 0, 1);
			}
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			// enable crystal ball transform here
			Vector4 vmin, vmax;
			g_scene->BBox(vmin, vmax);
			Vector4 centroid = (vmax + vmin) * 0.5;

			glTranslatef(0, 0, -g_dist_from_center / 100);
			glTranslatef(-g_scene_center_displacement[0], -g_scene_center_displacement[1], -g_scene_center_displacement[2]);
			Quaternion::setRotQuat(g_rotation_matrix, g_rotation_total_quat);
			glMultMatrixf(g_rotation_matrix.vals);
			glTranslatef(-centroid[0], -centroid[1], -centroid[2]);


			glGetDoublev(GL_MODELVIEW_MATRIX, g_modelview.vals);
			glGetDoublev(GL_PROJECTION_MATRIX, g_proj.vals);
			g_modelview = TransposeD(g_modelview);
			g_proj = TransposeD(g_proj);
			g_modproj = GMMMD(g_proj, g_modelview);

			glEnable(GL_DEPTH_TEST);


			// draw the scene
			//glTranslatef(- centroid[0], -centroid[1], -centroid[2]);

			if (g_draw_bounding_box) {
				Vector4 bgc;
				if (g_clear_color)
					bgc = { .05f, 0.05f, 0.05f, 1.0f };
				else
					bgc = { .9f, 0.9f, 0.9f, 1.0f };
				glBegin(GL_LINES);
				glColor3f(1, 0, 0);
				glVertex3f(vmin[0], vmin[1], vmin[2]);
				glColor3fv(bgc.vals);
				glVertex3f(vmax[0], vmin[1], vmin[2]);
				glVertex3f(vmin[0], vmax[1], vmin[2]);
				glVertex3f(vmax[0], vmax[1], vmin[2]);
				glVertex3f(vmin[0], vmin[1], vmax[2]);
				glVertex3f(vmax[0], vmin[1], vmax[2]);
				glVertex3f(vmin[0], vmax[1], vmax[2]);
				glVertex3f(vmax[0], vmax[1], vmax[2]);

				glColor3f(0, 1, 0);
				glVertex3f(vmin[0], vmin[1], vmin[2]);
				glColor3fv(bgc.vals);
				glVertex3f(vmin[0], vmax[1], vmin[2]);
				glVertex3f(vmax[0], vmin[1], vmin[2]);
				glVertex3f(vmax[0], vmax[1], vmin[2]);
				glVertex3f(vmin[0], vmin[1], vmax[2]);
				glVertex3f(vmin[0], vmax[1], vmax[2]);
				glVertex3f(vmax[0], vmin[1], vmax[2]);
				glVertex3f(vmax[0], vmax[1], vmax[2]);

				glColor3f(0, 0, 1);
				glVertex3f(vmin[0], vmin[1], vmin[2]);
				glColor3fv(bgc.vals);
				glVertex3f(vmin[0], vmin[1], vmax[2]);
				glVertex3f(vmax[0], vmin[1], vmin[2]);
				glVertex3f(vmax[0], vmin[1], vmax[2]);
				glVertex3f(vmin[0], vmax[1], vmin[2]);
				glVertex3f(vmin[0], vmax[1], vmax[2]);
				glVertex3f(vmax[0], vmax[1], vmin[2]);
				glVertex3f(vmax[0], vmax[1], vmax[2]);

				glEnd();
			}

			//glBegin(GL_POINTS);

			//glColor3f(0, 1, 0);
			//glVertex3f(vmin[0], vmin[1], vmin[2]);
			//glColor3f(0, 0, 1);
			//glVertex3f(centroid[0], centroid[1], centroid[2]);
			//glColor3f(1, 0, 0);
			//glVertex3f(vmax[0], vmax[1], vmax[2]);
			//glEnd();

			g_scene->Render();

			glutSwapBuffers();
		}


		void ViewerMouseFunction(int button, int state, int x, int y) {
			if (g_user_mouse_func != NULL && g_user_mouse_func(button, state, x, y)) return;
			mouseClicked(button, state, x, y);
		}
		
		void mousePassiveMovement(int mx, int my) {
			if (g_user_passive_mouse_func != NULL && g_user_passive_mouse_func(  mx, my)) return;
		}
		void mouseMovement(int mx, int my)
		{
			if (g_user_mouse_moved_func != NULL && g_user_mouse_moved_func(mx, my)) return;
			//printf("%d, %d\n", mx, my);
			Vector4 vmin, vmax;
			g_scene->BBox(vmin, vmax);
			Vector4 lengths = (vmax - vmin);
			int X = lengths[0];

		
			if (!g_mouseActive) return;
			//if (mx == 0 && my == 0) return;
			if (g_mouse_button_state == 2) {
				int dd = X;
				g_dist_from_center += dd*(my - g_mouse_y_old);
				g_mouse_y_old = my;

			}

			if (g_mouse_button_state == 1) {

				float ldim = (g_window_width > g_window_height ? g_window_width : g_window_height);


				g_screen_start_pos.vals[0] = g_mouse_x_old - .5 * g_window_width;
				g_screen_start_pos.vals[1] = .5 * g_window_height - g_mouse_y_old;

				if ((.5 * ldim) * (.5* ldim) - g_screen_start_pos.vals[0] * g_screen_start_pos.vals[0] - g_screen_start_pos.vals[1] * g_screen_start_pos.vals[1] <= 0) return;
				g_screen_start_pos.vals[2] = sqrt((.5 * ldim) * (.5* ldim) - g_screen_start_pos.vals[0] * g_screen_start_pos.vals[0] - g_screen_start_pos.vals[1] * g_screen_start_pos.vals[1]);
				Normalize3(&g_screen_start_pos);

				g_mouse_x_old = mx;
				g_mouse_y_old = my;

				// calculate the vectors;
				Vector4 vend;


				vend.vals[0] = mx - .5 * g_window_width;
				vend.vals[1] = .5 * g_window_height - my;

				if ((.5 * ldim) * (.5* ldim) - vend.vals[0] * vend.vals[0] - vend.vals[1] * vend.vals[1] <= 0) return;
				vend.vals[2] = sqrt((.5 * ldim) * (.5* ldim) - vend.vals[0] * vend.vals[0] - vend.vals[1] * vend.vals[1]);
				Normalize3(&vend);

				g_scene_center_displacement[0] += -lengths[0]* (vend.vals[0] - g_screen_start_pos.vals[0]);
				g_scene_center_displacement[1] += -lengths[0] * (vend.vals[1] - g_screen_start_pos.vals[1]);
				g_scene_center_displacement[2] += -0 * (vend.vals[2] - g_screen_start_pos.vals[2]);
				glutPostRedisplay();
			}

			if (g_mouse_button_state == 0) {
				float ldim = (g_window_width > g_window_height ? g_window_width : g_window_height);

				if (g_mouse_x_old == mx && g_mouse_y_old == my) return;

				g_screen_start_pos.vals[0] = g_mouse_x_old - .5 * g_window_width;
				g_screen_start_pos.vals[1] = .5 * g_window_height - g_mouse_y_old;

				if ((.5 * ldim) * (.5* ldim) - g_screen_start_pos.vals[0] * g_screen_start_pos.vals[0] - g_screen_start_pos.vals[1] * g_screen_start_pos.vals[1] < 0) return;
				g_screen_start_pos.vals[2] = sqrt((.5 * ldim) * (.5* ldim) - g_screen_start_pos.vals[0] * g_screen_start_pos.vals[0] - g_screen_start_pos.vals[1] * g_screen_start_pos.vals[1]);
				if (Normalize3(&g_screen_start_pos) == 0) return;
				//float xdiff = mx - g_mouse_x_old;
				//float ydiff = my - g_mouse_y_old;
				g_mouse_x_old = mx;
				g_mouse_y_old = my;

				// calculate the vectors;
				Vector4 vend;


				vend.vals[0] = mx - .5 * g_window_width;
				vend.vals[1] = .5 * g_window_height - my;

				if ((.5 * ldim) * (.5* ldim) - vend.vals[0] * vend.vals[0] - vend.vals[1] * vend.vals[1] <= 0) return;
				vend.vals[2] = sqrt((.5 * ldim) * (.5* ldim) - vend.vals[0] * vend.vals[0] - vend.vals[1] * vend.vals[1]);
				if (Normalize3(&vend) == 0) return;



				float alpha = InteriorAngle(g_screen_start_pos, vend);
				if (alpha < 0.01) return;

				Vector4 cp = Cross(g_screen_start_pos, vend);
				if (Normalize3(&cp) == 0) return;
				//printf("alpha: %f start: (%f, %f, %f)  end: (%f, %f, %f)  CP: (%f, %f, %f)\n", alpha, g_screen_start_pos.vals[0],g_screen_start_pos.vals[1],g_screen_start_pos.vals[2],vend.vals[0],vend.vals[1],vend.vals[2],cp.vals[0],cp.vals[1],cp.vals[2]);
				//update the crystal ball matrix
				glMatrixMode(GL_MODELVIEW);
				glLoadIdentity();

				//glMultMatrixf(g_base_view_matrix.vals);

				g_rotation_total_quat = Quaternion::times(g_rotation_total_quat, Quaternion::rotation(cp, alpha*PI / -180.0));

				Quaternion::setRotQuat(g_rotation_matrix, g_rotation_total_quat);
				glMultMatrixf(g_rotation_matrix.vals);
				//glRotatef(alpha, cp.vals[0], cp.vals[1], cp.vals[2]);
				//glMultMatrixf(g_base_view_matrix.vals);

				//glGetFloatv(GL_MODELVIEW_MATRIX, g_base_view_matrix.vals);

				// Redisplay image.
			}
			//g_numslices = 5;

			glutPostRedisplay();

		}

		// Respond to window resizing, preserving proportions.
			// Parameters give new window size in pixels.
			void reshapeMainWindow(int newWidth, int newHeight)
		{
			g_window_width = newWidth;
			g_window_height = newHeight;
			glViewport(0, 0, g_window_width, g_window_height);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			gluPerspective(g_field_of_view, GLfloat(g_window_width) / GLfloat(g_window_height), g_near_plane, g_far_plane);
			glutPostRedisplay();
		}

		void ViewerMouseMovedFunction(int mx, int my) {
			if (g_user_mouse_moved_func != NULL && g_user_mouse_moved_func(mx, my)) return;
			mouseMovement(mx, my);
		}

		void graphicKeys(unsigned char key, int x, int y)
		{



			switch (key)
			{
	
	

			case 'b':
				g_clear_color = !g_clear_color;
				printf("Background color is (%d, %d, %d)\n", g_clear_color, g_clear_color, g_clear_color);
				break;

			case 'R':
				render_view_2();
				break;
			case 'L':
				render_view_3();
				break;

			case 'd':
				render_view();
				break;
			case 'r':
				reset_view();
				break;

			case '.':
				g_draw_bounding_box = !g_draw_bounding_box;
				//rendertubes = !rendertubes;
				//printf("Render Tubes: %d\n",rendertubes);
				break;
			case 27:
				exit(0);
			default:
				break;
				//cout << key << endl;
			}
			glutPostRedisplay();
		}
		void ViewerGraphicKeyFunction(unsigned char key, int x, int y) {
			if (g_user_graphic_keys_func != NULL && g_user_graphic_keys_func(key, x, y)) return;
			graphicKeys(key, x, y);
		}
	}



	void SetScene(RenderableGL* scene) {
		INTERNAL::g_scene = scene;
	}

	void SetMouseFunction(bool(*callback)(int, int, int, int)) {
		INTERNAL::g_user_mouse_func = callback;
	}

	void SetGraphicKeysFunction(bool(*callback)(unsigned char, int, int)) {
		INTERNAL::g_user_graphic_keys_func = callback;
	}

	void SetPassiveMotionFunc(bool(*func)(int , int )) {
		INTERNAL::g_user_passive_mouse_func = func;
	}

	void SetMouseMovedFunction(bool(*callback)(int, int)) {
		INTERNAL::g_user_mouse_moved_func = callback;
	}
	void SetTimerFunction(unsigned int time, void(*callback)(int), int value) {
		glutTimerFunc(time, callback, value);
	}

	void Initialize_GLUT(int argc, char **argv) {

		INTERNAL::g_scene_center_displacement[0] = 0.0;
		INTERNAL::g_scene_center_displacement[1] = 0.0;
		INTERNAL::g_scene_center_displacement[2] = 0.0;
		const int WIDTH = 800;
		const int HEIGHT = 600;

		// Current size of window.
		int g_window_width = WIDTH;
		int g_window_height = HEIGHT;
		// Mouse positions, normalized to [0,1].
		double xMouse = 0.0;
		double yMouse = 0.0;
		INTERNAL::g_mouseActive = false;

		// GLUT initialization.
		int noargs = 0;
		glutInit(&noargs, NULL);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		glutInitWindowSize(g_window_width, g_window_height);
		glutInitWindowPosition(100, 80);
		INTERNAL::g_main_window = glutCreateWindow("Morse3d Efficient Computation");

		glewInit();

		glutDisplayFunc(INTERNAL::display);
		glutReshapeFunc(INTERNAL::reshapeMainWindow);
		glutKeyboardFunc(INTERNAL::ViewerGraphicKeyFunction);
		glutMouseFunc(INTERNAL::mouseClicked);
		glutMotionFunc(INTERNAL::mouseMovement);
		glutPassiveMotionFunc(INTERNAL::mousePassiveMovement);

		glDepthMask(true);
		glClearColor(0, 0, 0, 1);

		glDisable(GL_CULL_FACE);
		glDepthFunc(GL_LESS);
		glEnable(GL_DEPTH_TEST);

		glShadeModel(GL_SMOOTH);
		glLineWidth(1);
		glEnable(GL_LINE_SMOOTH);

		// Enable our light.
		glLightfv(GL_LIGHT0, GL_DIFFUSE, INTERNAL::g_diffuse_light);
		glLightfv(GL_LIGHT0, GL_POSITION, INTERNAL::g_light_position);
		glEnable(GL_LIGHT0);

		// Set up the material information for our objects.  Again this is just for show.
		glEnable(GL_COLOR_MATERIAL);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, INTERNAL::g_specular_light);
		glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 128);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

	}
	void StartGlutMainLoop() {

		if (INTERNAL::g_scene == NULL) {
			printf("Scene not initialized, exiting\n");
			exit(0);
		}

		// now its safe to initialize view of scene
		// initialize crystal ball
		INTERNAL::reset_view();

		glutMainLoop();
	}
	void Redisplay() {
		glutPostRedisplay();
	}
}